# scripts/reward_probe.py
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.wrappers import make_crazyclimber_env
from src.networks import QNetwork

def to_chw(obs):
    arr = np.array(obs)
    if arr.shape == (4, 84, 84):
        return arr
    if arr.shape[-1] == 4:
        return np.transpose(arr, (2, 0, 1))
    raise ValueError(f"Unexpected obs shape {arr.shape}")

def greedy_action(qnet, state_chw, device):
    with torch.no_grad():
        s = torch.from_numpy(state_chw[None]).float().to(device)
        q = qnet(s)
        return int(torch.argmax(q, dim=1).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max_steps", type=int, default=3000)
    ap.add_argument("--policy", choices=["random","greedy"], default="random")
    ap.add_argument("--model", type=str, default="runs/baseline/qnet.pt")
    ap.add_argument("--out_csv", type=str, default="runs/baseline/reward_probe.csv")
    ap.add_argument("--summary_json", type=str, default="runs/baseline/reward_summary.json")
    args = ap.parse_args()

    env = make_crazyclimber_env(seed=123, render_mode=None)
    n_actions = env.action_space.n

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    qnet = None
    if args.policy == "greedy":
        qnet = QNetwork(n_actions).to(device)
        qnet.load_state_dict(torch.load(args.model, map_location=device))
        qnet.eval()

    out_path = ROOT / args.out_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)

    totals = []
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode","step","reward","cum_return"])
        for ep in range(1, args.episodes+1):
            obs, _ = env.reset()
            state = to_chw(obs)
            cum = 0.0
            for t in range(args.max_steps):
                if args.policy == "random":
                    a = env.action_space.sample()
                else:
                    a = greedy_action(qnet, state, device)
                next_obs, r, terminated, truncated, _ = env.step(a)
                cum += float(r)
                w.writerow([ep, t+1, float(r), cum])
                state = to_chw(next_obs)
                if terminated or truncated:
                    break
            totals.append({"episode": ep, "return": cum, "steps": t+1})

    env.close()
    summary = {
        "policy": args.policy,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "totals": totals
    }
    (ROOT / args.summary_json).write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {ROOT / args.summary_json}")

if __name__ == "__main__":
    main()
