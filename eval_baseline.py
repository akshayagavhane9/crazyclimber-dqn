# eval_baseline.py
import csv
import numpy as np
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", default="runs/baseline/metrics.csv")
    args = ap.parse_args()

    rows = []
    with open(args.metrics_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    returns = np.array([float(x["return"]) for x in rows], dtype=np.float32)
    steps   = np.array([int(x["steps"]) for x in rows], dtype=np.int32)
    eps     = np.array([float(x["epsilon"]) for x in rows], dtype=np.float32)

    print(f"Episodes: {len(rows)}")
    print(f"Return: mean={returns.mean():.2f} median={np.median(returns):.2f} "
          f"last10_avg={returns[-10:].mean():.2f}")
    print(f"Steps:  mean={steps.mean():.1f}  median={np.median(steps):.1f}")
    print(f"Epsilon: first={eps[0]:.3f} last={eps[-1]:.3f}")

if __name__ == "__main__":
    main()
