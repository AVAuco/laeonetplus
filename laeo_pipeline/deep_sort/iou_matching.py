"""IoU-based cost matrix, used as a fallback matching stage for unconfirmed tracks
and tracks that appearance matching failed to associate."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def iou(bbox: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """`bbox`, each row of `candidates`: [x, y, w, h]."""
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[
        np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
        np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
    ]
    br = np.c_[
        np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
        np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
    ]
    wh = np.maximum(0.0, br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices: Optional[List[int]] = None,
             detection_indices: Optional[List[int]] = None) -> np.ndarray:
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        if track.time_since_update > 1:
            cost_matrix[row, :] = 1e5
            continue

        bbox = track.to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix
