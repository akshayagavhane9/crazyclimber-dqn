# src/networks.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """
    Input: (B, C=4, H=84, W=84)  # 4 stacked grayscale frames
    Output: (B, num_actions)
    """
    def __init__(self, num_actions: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # -> (32, 20, 20)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # -> (64, 9, 9)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> (64, 7, 7)
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect NCHW (B,4,84,84). If NHWC provided, swap dims in the caller.
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
