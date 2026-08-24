"""Nearest-neighbor appearance distance metric with a per-track feature gallery."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return 1.0 - a @ b.T


class NearestNeighborDistanceMetric:
    def __init__(self, matching_threshold: float, budget: Optional[int] = None) -> None:
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: Dict[int, List[np.ndarray]] = {}

    def partial_fit(self, features: np.ndarray, targets: np.ndarray, active_targets: List[int]) -> None:
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets if k in self.samples}

    def distance(self, features: np.ndarray, targets: List[int]) -> np.ndarray:
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            gallery = np.asarray(self.samples[target])
            cost_matrix[i, :] = _cosine_distance(gallery, features).min(axis=0)
        return cost_matrix
