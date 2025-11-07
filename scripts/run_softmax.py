# scripts/run_softmax.py
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

# --- Project imports ---
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.wrappers import make_crazyclimber_env
from src.networks import QNetwork


# ---------- Utils ----------
def to_chw(obs):
    """
    Convert env obs to (C,H,W) = (4,84,84).
    Env may return (4,84,84) or (84,84,4) LazyFrames.
    """
    arr = np.array(obs)
    if arr.shape == (4, 84, 84):
        return arr
    if arr.shape[-1] == 4:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unexpected obs shape {arr.shape}")


def softmax_action(q_values: torch.Tensor, temp: float = 1.0) -> int:
    """
    Numerically-stable Boltzmann exploration over Q-values.
    - Falls back to greedy when temp is tiny
    - Uses PyTorch softmax (stable) + final safety renorm.
    """
    if temp <= 1e-6:
        return int(torch.argmax(q_values).item())

    probs = F.softmax(q_values / temp, dim=-1).detach().cpu().numpy()
    probs = probs.astype(np.float64)
    # Safety: clip tiny values and renormalize to sum=1
    probs = np.clip(probs, 1e-12, 1.0)
    probs = probs / probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ---------- Training loop (one-step TD, no replay) ----------
def train_softmax(
    out_dir: Path,
    episodes: int = 50,
    max_steps: int = 3000,
    lr: float = 1e-4,
    gamma: float = 0.99,
    temp: float = 1.0,
    seed: int = 123,
):
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print("Device:", device)
    print(f"Running Softmax policy with temp={temp}")

    env = make_crazyclimber_env(seed=seed, render_mode=None)
    n_actions = env.action_space.n

    qnet = QNetwork(n_actions).to(device)
    optimizer = torch.optim.Adam(qnet.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()

    episode_returns = []

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state = to_chw(obs)
        done = False
        total_r = 0.0
        step = 0

        while not done and step < max_steps:
            s_t = torch.from_numpy(state[None]).float().to(device)
            q_values = qnet(s_t)[0]  # shape [n_actions]

            # Softmax / Boltzmann action
            a = softmax_action(q_values, temp=temp)

            next_obs, r, terminated, truncated, _ = env.step(a)
            done = bool(terminated or truncated)
            next_state = to_chw(next_obs)

            # One-step TD target
            with torch.no_grad():
                next_q = qnet(torch.from_numpy(next_state[None]).float().to(device))[0]
                target = r + gamma * next_q.max().item() * (0.0 if done else 1.0)

            pred = q_values[a]
            loss = loss_fn(pred, torch.tensor(target, dtype=torch.float32, device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_r += float(r)
            step += 1

        episode_returns.append(total_r)
        print(f"[Ep {ep:2d}/{episodes}] return={total_r:.1f} steps={step}")

    env.close()

    out_dir.mkdir(parents=True, exist_ok=True)
    # Save returns as float64 to be safe when reading with numpy
    np.save(out_dir / "returns.npy", np.array(episode_returns, dtype=np.float64))
    # Also save a tiny config for reproducibility
    (out_dir / "config.json").write_text(
        (
            "{\n"
            f'  "policy": "softmax",\n'
            f'  "temp": {temp},\n'
            f'  "gamma": {gamma},\n'
            f'  "lr": {lr},\n'
            f'  "episodes": {episodes},\n'
            f'  "max_steps": {max_steps},\n'
            f'  "seed": {seed}\n'
            "}\n"
        )
    )
    print(f"Saved: {out_dir}")


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--temp", type=float, default=1.0, help="Softmax temperature (higher = more exploration)")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out_dir", type=str, default="runs/softmax_temp_1_0")
    args = ap.parse_args()

    train_softmax(
        out_dir=Path(args.out_dir),
        episodes=args.episodes,
        max_steps=args.max_steps,
        lr=args.lr,
        gamma=args.gamma,
        temp=args.temp,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
