"""Hungarian-algorithm matching and the DeepSORT matching cascade."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import kalman_filter as kf_module

INFTY_COST = 1e5


def min_cost_matching(
    distance_metric: Callable,
    max_distance: float,
    tracks,
    detections,
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    if not detection_indices or not track_indices:
        return [], track_indices, detection_indices

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric: Callable,
    max_distance: float,
    cascade_depth: int,
    tracks,
    detections,
    track_indices: Optional[List[int]] = None,
    detection_indices: Optional[List[int]] = None,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if not unmatched_detections:
            break

        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]
        if not track_indices_l:
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric, max_distance, tracks, detections, track_indices_l, unmatched_detections
        )
        matches += matches_l

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
    kf,
    cost_matrix: np.ndarray,
    tracks,
    detections,
    track_indices: List[int],
    detection_indices: List[int],
    gated_cost: float = INFTY_COST,
    only_position: bool = False,
) -> np.ndarray:
    gating_dim = 2 if only_position else 4
    gating_threshold = kf_module.CHI2_INV_95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
