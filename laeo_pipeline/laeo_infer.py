"""Runs the PyTorch LAEO-Net++ model on the temporal samples built by `PairBuffer`."""

from __future__ import annotations

import numpy as np
import torch

from .laeo_model import load_laeo_model
from .pair_buffer import PairSample


class LaeoScorer:
    def __init__(self, json_path: str, weights_path: str, mean_map_path: str, device: str = "cpu") -> None:
        self.device = device
        self.model = load_laeo_model(json_path, weights_path, device=device)

        mean_map_flat = np.load(mean_map_path)  # shape: (window_len * side, side, 3)
        side = mean_map_flat.shape[1]
        window_len = mean_map_flat.shape[0] // side
        self.mean_map = mean_map_flat.reshape(window_len, side, side, 3).astype(np.float32)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return t.permute(3, 0, 1, 2).unsqueeze(0)  # (T,H,W,3) -> (1,3,T,H,W)

    @torch.no_grad()
    def score(self, sample: PairSample) -> float:
        mean_map = self.mean_map
        window_len = sample.maps.shape[0]
        if window_len != mean_map.shape[0]:
            idx = np.linspace(0, mean_map.shape[0] - 1, window_len).astype(int)
            mean_map = mean_map[idx]

        head_left = sample.crops_left.astype(np.float32) / 255.0
        head_right = sample.crops_right.astype(np.float32) / 255.0
        map_norm = (sample.maps.astype(np.float32) - mean_map) / 255.0

        probs = self.model(
            self._to_tensor(head_left), self._to_tensor(head_right), self._to_tensor(map_norm)
        )
        return float(probs[0, 1].item())
