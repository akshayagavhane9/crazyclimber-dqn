# src/replay_buffer.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: tuple, dtype=np.float32):
        self.capacity = capacity
        self.ptr = 0
        self.full = False

        self.states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=dtype)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = done

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        max_i = self.capacity if self.full else self.ptr
        idx = rng.integers(0, max_i, size=batch_size)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
