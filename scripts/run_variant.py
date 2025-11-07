# scripts/run_variant.py
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from train_dqn import run_baseline  # re-use the trainer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--eps_start", type=float, default=1.0)
    ap.add_argument("--eps_end", type=float, default=0.05)
    ap.add_argument("--eps_decay_episodes", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_baseline(
        total_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_episodes=args.eps_decay_episodes,
        out_dir=args.out_dir,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
