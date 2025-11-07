# src/wrappers.py
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

def make_crazyclimber_env(seed: int | None = 42, render_mode: str | None = None):
    """
    Returns ALE/CrazyClimber-v5 with standard DQN preprocessing:
      - grayscale 84x84
      - v5 env already has frame-skip=4 → set frame_skip=1 here to avoid double skipping
      - frame stack=4
    """
    env = gym.make("ALE/CrazyClimber-v5", render_mode=render_mode)

    # IMPORTANT: frame_skip=1 because ALE v5 already uses frame-skip (default 4).
    env = AtariPreprocessing(
        env,
        grayscale_obs=True,
        screen_size=84,
        scale_obs=True,   # float32 in [0,1]
        frame_skip=1      # <-- prevent double frame-skipping
    )

    # Stack the last 4 frames
    env = FrameStack(env, num_stack=4)

    if seed is not None:
        env.reset(seed=seed)

    return env
