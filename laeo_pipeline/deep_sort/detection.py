"""Detection container consumed by the DeepSORT tracker."""

from __future__ import annotations

from typing import Optional

import numpy as np


class Detection:
    """A single bounding-box detection in a frame.

    `tlwh`: [x, y, w, h] (top-left corner + size).
    `confidence`: detector score.
    `feature`: optional appearance descriptor (1D array) used for re-identification.
    """

    def __init__(self, tlwh, confidence: float, feature: Optional[np.ndarray] = None) -> None:
        self.tlwh = np.asarray(tlwh, dtype=np.float32)
        self.confidence = float(confidence)
        self.feature = None if feature is None else np.asarray(feature, dtype=np.float32)

    def to_tlbr(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self) -> np.ndarray:
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
