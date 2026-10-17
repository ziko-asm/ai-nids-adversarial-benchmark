"""Detectors compared in the benchmark.

mlp  - 3-layer fully connected network.
cnn  - 1D CNN over the feature vector. Tabular features have no natural order,
       so the convolution has no built-in advantage here; testing whether it
       is nonetheless more robust is one of the benchmark's questions.
gbt  - gradient-boosted trees (scikit-learn). Not differentiable, so it is only
       attacked by transferring adversarial examples crafted on a neural net.
"""

from __future__ import annotations

import torch
from torch import nn

NEURAL_MODELS = ("mlp", "cnn")


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, in_dim: int, channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, channels // 2, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(16),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(channels * 16, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.unsqueeze(1)))


def build_model(name: str, in_dim: int) -> nn.Module:
    if name == "mlp":
        return MLP(in_dim)
    if name == "cnn":
        return CNN1D(in_dim)
    raise ValueError(f"Unknown model '{name}'. Choose one of {NEURAL_MODELS}")
