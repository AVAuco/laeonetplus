"""OpenCV drawing helpers for the LAEO pipeline output video."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import cv2
import numpy as np

GREEN = (0, 200, 0)
RED = (0, 0, 220)
BOX_COLOR = (255, 200, 0)
LINE_COLOR_NEUTRAL = (0, 200, 255)


def _bbox_center(bbox: Sequence[float]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def draw_heads(frame: np.ndarray, tracks: Dict[int, Sequence[float]]) -> None:
    """Draw a bounding box + track id label for every currently visible track."""
    for track_id, bbox in tracks.items():
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        label = f"ID {track_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(frame, label, (x1 + 2, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_pair(frame: np.ndarray, bbox_a: Sequence[float], bbox_b: Sequence[float],
              probability: float, threshold: float) -> None:
    """Draw the connecting line between a pair of heads and its LAEO probability."""
    color = GREEN if probability >= threshold else RED
    ca = _bbox_center(bbox_a)
    cb = _bbox_center(bbox_b)

    cv2.line(frame, ca, cb, color, 2)

    mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
    label = f"{probability * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (mid[0] - tw // 2 - 3, mid[1] - th - 6), (mid[0] + tw // 2 + 3, mid[1] + 2), (0, 0, 0), -1)
    cv2.putText(frame, label, (mid[0] - tw // 2, mid[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_frame_index(frame: np.ndarray, frame_idx: int) -> None:
    cv2.putText(frame, f"frame {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
