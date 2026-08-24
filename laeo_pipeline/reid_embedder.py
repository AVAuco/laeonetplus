"""
Lightweight PyTorch appearance embedder for the DeepSORT tracker.

It is a small, fixed (frozen, deterministically-seeded) convolutional network that
maps a head crop to a compact L2-normalized descriptor used for cosine-distance
appearance matching. It is not trained for re-identification -- for short head
tracks the Kalman motion model and IoU already do most of the work, and this
descriptor mainly helps disambiguate two heads that briefly overlap or occlude one
another -- but it keeps the tracker fully self-contained and independent of any
extra pretrained-weights download.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import nn


class _EmbedderNet(nn.Module):
    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.project = nn.Linear(128, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.project(x)
        return torch.nn.functional.normalize(x, p=2.0, dim=1)


class ReIDEmbedder:
    def __init__(self, device: str = "cpu", embedding_dim: int = 128, seed: int = 1330) -> None:
        self.device = device
        generator = torch.Generator().manual_seed(seed)
        self.net = _EmbedderNet(embedding_dim)
        with torch.no_grad():
            for p in self.net.parameters():
                if p.dim() > 1:
                    nn.init.kaiming_normal_(p, generator=generator)
                else:
                    nn.init.zeros_(p)
        self.net.to(device)
        self.net.eval()

    @torch.no_grad()
    def embed(self, crops: List[np.ndarray]) -> np.ndarray:
        """`crops`: list of BGR uint8 arrays (any size). Returns (N, embedding_dim) float32."""
        if not crops:
            return np.zeros((0, self.net.project.out_features), dtype=np.float32)

        import cv2

        batch = np.stack([cv2.resize(c, (64, 64)) for c in crops]).astype(np.float32) / 255.0
        batch = batch[..., ::-1].copy()  # BGR -> RGB
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        out = self.net(tensor)
        return out.cpu().numpy()
