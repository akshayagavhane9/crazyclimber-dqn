# train_dqn.py
from __future__ import annotations
import os, time, json, csv
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.wrappers import make_crazyclimber_env
from src.networks import QNetwork
from src.replay_buffer import ReplayBuffer

def to_chw(obs_np: np.ndarray) -> np.ndarray:
    # Accept (4,84,84) or (84,84,4) → return (4,84,84)
    if obs_np.shape == (4, 84, 84):
        return obs_np
    if obs_np.shape == (84, 84, 4):
        return np.transpose(obs_np, (2, 0, 1))
    raise ValueError(f"Unexpected obs shape {obs_np.shape}")

def select_action(qnet: QNetwork, state_chw: np.ndarray, epsilon: float, device) -> int:
    if np.random.rand() < epsilon:
        return np.random.randint(0, qnet.fc[-1].out_features)
    with torch.no_grad():
        s = torch.from_numpy(state_chw[None]).float().to(device)  # (1,4,84,84)
        q = qnet(s)  # (1, A)
        return int(torch.argmax(q, dim=1).item())

def soft_update(target: nn.Module, online: nn.Module, tau: float = 0.005):
    with torch.no_grad():
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.lerp_(op.data, tau)

def hard_update(target: nn.Module, online: nn.Module):
    target.load_state_dict(online.state_dict())

def train_step(qnet, target, optimizer, batch, device, gamma: float):
    states, actions, rewards, next_states, dones = batch
    # Convert to torch
    states_t = torch.from_numpy(states).float().to(device)          # (B,4,84,84)
    actions_t = torch.from_numpy(actions).long().to(device)         # (B,)
    rewards_t = torch.from_numpy(rewards).float().to(device)        # (B,)
    next_states_t = torch.from_numpy(next_states).float().to(device)# (B,4,84,84)
    dones_t = torch.from_numpy(dones.astype(np.float32)).to(device) # (B,)

    # Q(s,a)
    q_pred = qnet(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        # Max_a' Q_target(s', a')
        q_next = target(next_states_t).max(dim=1).values
        target_q = rewards_t + gamma * (1.0 - dones_t) * q_next

    loss = nn.SmoothL1Loss()(q_pred, target_q)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(qnet.parameters(), max_norm=10.0)
    optimizer.step()

    return float(loss.item())

def run_baseline(
    total_episodes=50,
    max_steps_per_ep=3000,
    buffer_capacity=50_000,
    warmup_steps=10_000,
    batch_size=64,
    gamma=0.99,
    lr=1e-4,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay_episodes=40,
    target_update="soft",   # "soft" or "hard"
    tau=0.005,              # for soft
    hard_update_interval=1000,  # steps
    seed=42,
    out_dir="runs/baseline",
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    env = make_crazyclimber_env(seed=seed, render_mode=None)
    n_actions = env.action_space.n

    # Peek obs shape for buffer
    obs, info = env.reset(seed=seed)
    obs_chw = to_chw(np.array(obs))
    obs_shape = obs_chw.shape  # (4,84,84)

    # Networks
    qnet = QNetwork(n_actions).to(device)
    target = QNetwork(n_actions).to(device)
    hard_update(target, qnet)

    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    rb = ReplayBuffer(buffer_capacity, obs_shape, dtype=np.float32)

    # Warmup with random actions (fills replay)
    print(f"Warmup for {warmup_steps} steps...")
    state = obs_chw
    steps_done = 0
    while steps_done < warmup_steps:
        a = env.action_space.sample()
        next_obs, r, terminated, truncated, _ = env.step(a)
        next_chw = to_chw(np.array(next_obs))
        done = terminated or truncated
        rb.add(state, a, r, next_chw, done)
        state = next_chw if not done else to_chw(np.array(env.reset()[0]))
        steps_done += 1

    # Training
    returns = []
    steps_per_ep = []
    epsilons = []
    loss_moving = deque(maxlen=200)

    global_step = 0
    for ep in range(1, total_episodes + 1):
        # Linear epsilon decay over eps_decay_episodes
        epsilon = max(
            eps_end,
            eps_start - (eps_start - eps_end) * (ep - 1) / max(1, eps_decay_episodes)
        )
        epsilons.append(float(epsilon))

        obs, _ = env.reset()
        state = to_chw(np.array(obs))
        ep_return = 0.0
        ep_steps = 0

        for t in range(max_steps_per_ep):
            a = select_action(qnet, state, epsilon, device)
            next_obs, r, terminated, truncated, _ = env.step(a)
            next_state = to_chw(np.array(next_obs))
            done = terminated or truncated

            rb.add(state, a, r, next_state, done)
            state = next_state
            ep_return += r
            ep_steps += 1
            global_step += 1

            # Sample & learn
            if len(rb) >= batch_size:
                batch = rb.sample(batch_size)
                loss = train_step(qnet, target, optimizer, batch, device, gamma)
                loss_moving.append(loss)

                # Target updates
                if target_update == "soft":
                    soft_update(target, qnet, tau=tau)
                else:
                    if global_step % hard_update_interval == 0:
                        hard_update(target, qnet)

            if done:
                break

        returns.append(float(ep_return))
        steps_per_ep.append(int(ep_steps))

        if ep % 5 == 0 or ep == 1:
            print(f"[Ep {ep}/{total_episodes}] return={ep_return:.1f} steps={ep_steps} "
                  f"eps={epsilon:.3f} loss~={np.mean(loss_moving) if loss_moving else 0:.4f}")

    env.close()

    # Persist metrics
    metrics_path = os.path.join(out_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "steps", "epsilon"])
        for i, (r, s, e) in enumerate(zip(returns, steps_per_ep, epsilons), start=1):
            w.writerow([i, r, s, e])

    conf = dict(
        total_episodes=total_episodes,
        max_steps_per_ep=max_steps_per_ep,
        buffer_capacity=buffer_capacity,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
        gamma=gamma,
        lr=lr,
        eps_start=eps_start,
        eps_end=eps_end,
        eps_decay_episodes=eps_decay_episodes,
        target_update=target_update,
        tau=tau,
        hard_update_interval=hard_update_interval,
        seed=seed,
        device=str(device),
    )
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(conf, f, indent=2)

    torch.save(qnet.state_dict(), os.path.join(out_dir, "qnet.pt"))
    print(f"\nSaved: {out_dir}")
    print(f"Avg return (last 10): {np.mean(returns[-10:]) if len(returns)>=10 else np.mean(returns):.2f}")
    print(f"Avg steps  (last 10): {np.mean(steps_per_ep[-10:]) if len(steps_per_ep)>=10 else np.mean(steps_per_ep):.1f}")

if __name__ == "__main__":
    # Baseline defaults: short run to verify pipeline
    run_baseline(
        total_episodes=50,      # use 100 later for comparison runs
        max_steps_per_ep=3000,  # Atari needs more than 99 steps
        warmup_steps=10_000,
        eps_decay_episodes=40,
        out_dir="runs/baseline",
    )
