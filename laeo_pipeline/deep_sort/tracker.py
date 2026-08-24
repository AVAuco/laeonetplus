"""DeepSORT multi-object tracker: Kalman motion model + cascaded appearance/IoU matching."""

from __future__ import annotations

from typing import List

import numpy as np

from . import iou_matching, linear_assignment
from .kalman_filter import KalmanFilter
from .nn_matching import NearestNeighborDistanceMetric
from .track import Track


class Tracker:
    def __init__(
        self,
        max_cosine_distance: float = 0.3,
        nn_budget: int = 100,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
    ) -> None:
        self.metric = NearestNeighborDistanceMetric(max_cosine_distance, nn_budget)
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks: List[Track] = []
        self._next_id = 1

    def predict(self) -> None:
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections) -> None:
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id] * len(track.features)
            track.features = []
        if features:
            self.metric.partial_fit(np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.asarray([dets[i].feature for i in detection_indices])
            targets = [tracks[i].track_id for i in track_indices]
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices
            )
            return cost_matrix

        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        candidates_b = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            candidates_b,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection) -> None:
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, covariance, self._next_id, self.n_init, self.max_age, detection.feature)
        )
        self._next_id += 1
