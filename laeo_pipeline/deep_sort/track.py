"""Single-target track state, following the DeepSORT track lifecycle."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

import numpy as np


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


class Track:
    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        feature: Optional[np.ndarray] = None,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.TENTATIVE
        self.features: List[np.ndarray] = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self) -> np.ndarray:
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]  # aspect * height -> width
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self) -> np.ndarray:
        ret = self.to_tlwh()
        ret[2:] += ret[:2]
        return ret

    def predict(self, kf) -> None:
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection) -> None:
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        if detection.feature is not None:
            self.features.append(detection.feature)
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.TENTATIVE and self.hits >= self._n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.DELETED
        elif self.time_since_update > self._max_age:
            self.state = TrackState.DELETED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
