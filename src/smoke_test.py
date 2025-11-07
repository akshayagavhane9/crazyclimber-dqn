# src/smoke_test.py
import numpy as np
import torch

from src.wrappers import make_crazyclimber_env
from src.networks import QNetwork

def ensure_chw(x: np.ndarray) -> np.ndarray:
    """
    Ensure observation is channels-first (C,H,W) with C=4.
    Accepts either (4,84,84) or (84,84,4).
    """
    if x.ndim != 3:
        raise ValueError(f"Expected 3D obs, got shape {x.shape}")
    # Already CHW
    if x.shape[0] == 4:
        return x
    # HWC -> CHW
    if x.shape[-1] == 4:
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"Unexpected obs shape {x.shape}; cannot find channel dim of 4")

def main():
    env = make_crazyclimber_env(seed=123, render_mode=None)
    obs, info = env.reset()

    obs_np = np.array(obs)  # LazyFrames -> ndarray
    print("Initial obs shape:", obs_np.shape)  # expect (4,84,84) or (84,84,4)

    n_actions = env.action_space.n
    print("Action space:", env.action_space)
    print("Number of actions:", n_actions)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    qnet = QNetwork(num_actions=n_actions).to(device)

    # Step a few times just to get a recent frame
    last_obs = obs_np
    for _ in range(4):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        last_obs = np.array(next_obs)
        if terminated or truncated:
            obs, info = env.reset()
            last_obs = np.array(obs)

    chw = ensure_chw(last_obs)             # (4,84,84) channels-first
    batch = np.stack([chw, chw], axis=0)   # (B=2,4,84,84)

    with torch.no_grad():
        logits = qnet(torch.from_numpy(batch).float().to(device))
    print("QNetwork output shape:", tuple(logits.shape))  # expect (2, 9)

    env.close()
    print("Smoke test passed ✔")

if __name__ == "__main__":
    main()
