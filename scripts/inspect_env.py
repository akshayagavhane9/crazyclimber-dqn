# scripts/inspect_env.py
from __future__ import annotations
import numpy as np
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))          # allow `from src...` imports
sys.path.append(str(ROOT / "src"))

from src.wrappers import make_crazyclimber_env

def ensure_chw(x: np.ndarray) -> np.ndarray:
    if x.shape == (4, 84, 84):
        return x
    if x.shape[-1] == 4:
        return np.transpose(x, (2, 0, 1))
    raise ValueError(f"Unexpected obs shape {x.shape}")

def to_py(obj):
    """Recursively convert NumPy types -> native Python types for JSON dumping."""
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.dtype,)):
        return str(obj)
    return obj

def main(out_json: Path):
    env = make_crazyclimber_env(seed=123, render_mode=None)
    obs, info = env.reset()
    obs_np = np.array(obs)
    chw = ensure_chw(obs_np)

    # Observation space (post-wrappers)
    obs_info = {
        "shape": [int(chw.shape[0]), int(chw.shape[1]), int(chw.shape[2])],
        "dtype": str(chw.dtype),                 # e.g., float32
        "min": float(chw.min()),
        "max": float(chw.max()),
        "stack_size": int(chw.shape[0]),
        "resolution": [int(chw.shape[1]), int(chw.shape[2])],
        "scaled_0_1": True
    }

    # Action space & meanings
    n_actions = int(env.action_space.n)
    try:
        meanings = env.unwrapped.get_action_meanings()
    except Exception:
        meanings = [f"A{i}" for i in range(n_actions)]

    actions = [{"id": int(i), "meaning": str(meanings[i]) if i < len(meanings) else f"A{i}"} 
               for i in range(n_actions)]

    summary = {
        "env_id": "ALE/CrazyClimber-v5",
        "action_space_n": n_actions,
        "actions": actions,
        "observation": obs_info,
        "notes": {
            "frameskip_v5": 4,
            "wrapper_frame_skip": 1,
            "framestack": 4,
            "grayscale_84x84": True
        }
    }

    env.close()
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Convert any remaining NumPy values to Python before dumping
    summary_py = to_py(summary)
    out_json.write_text(json.dumps(summary_py, indent=2))
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    OUT = Path(__file__).resolve().parents[1] / "runs" / "baseline" / "env_analysis.json"
    main(OUT)
