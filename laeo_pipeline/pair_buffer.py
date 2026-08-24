"""
Sliding-window buffering of per-track head crops, used to assemble the temporal
inputs (two head-crop sequences + head-position map sequence) that LAEO-Net++
expects for every candidate pair of currently-tracked heads.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .geometry import BBox, crop_head, draw_pair_head_map


class _TrackHistory:
    def __init__(self, maxlen: int) -> None:
        self.frames: deque = deque(maxlen=maxlen)

    def push(self, frame_idx: int, bbox: BBox, crop: np.ndarray) -> None:
        if self.frames and self.frames[-1][0] != frame_idx - 1:
            self.frames.clear()
        self.frames.append((frame_idx, bbox, crop))

    def is_full(self, window_len: int) -> bool:
        return len(self.frames) == window_len


class PairSample:
    def __init__(self, track_a: int, track_b: int, crops_left: np.ndarray,
                 crops_right: np.ndarray, maps: np.ndarray) -> None:
        self.track_a = track_a
        self.track_b = track_b
        self.crops_left = crops_left
        self.crops_right = crops_right
        self.maps = maps

    @property
    def pair_key(self) -> Tuple[int, int]:
        return tuple(sorted((self.track_a, self.track_b)))


class PairBuffer:
    def __init__(self, frame_shape: Sequence[int], window_len: int = 10,
                 head_size: int = 64, map_size: Tuple[int, int] = (64, 64)) -> None:
        self.frame_shape = frame_shape
        self.window_len = window_len
        self.head_size = head_size
        self.map_size = map_size
        self.histories: Dict[int, _TrackHistory] = {}

    def remove_track(self, track_id: int) -> None:
        self.histories.pop(track_id, None)

    def update(self, frame_idx: int, frame: np.ndarray, tracks: List[Tuple[int, BBox]]) -> List[PairSample]:
        active_ids = []
        for track_id, bbox in tracks:
            active_ids.append(track_id)
            hist = self.histories.setdefault(track_id, _TrackHistory(self.window_len))
            crop = crop_head(frame, bbox, self.head_size)
            hist.push(frame_idx, bbox, crop)

        active_ids.sort()
        ready: List[PairSample] = []
        for i in range(len(active_ids)):
            for j in range(i + 1, len(active_ids)):
                id_a, id_b = active_ids[i], active_ids[j]
                hist_a, hist_b = self.histories[id_a], self.histories[id_b]
                if hist_a.is_full(self.window_len) and hist_b.is_full(self.window_len):
                    ready.append(self._build_sample(id_a, hist_a, id_b, hist_b))
        return ready

    def _build_sample(self, id_a: int, hist_a: _TrackHistory, id_b: int, hist_b: _TrackHistory) -> PairSample:
        frames_a = list(hist_a.frames)
        frames_b = list(hist_b.frames)

        # Fix a consistent left/right assignment for the whole window, based on the
        # horizontal position at the start of the window.
        _, bbox_a0, _ = frames_a[0]
        _, bbox_b0, _ = frames_b[0]
        if bbox_a0[0] <= bbox_b0[0]:
            left_id, left_frames, right_id, right_frames = id_a, frames_a, id_b, frames_b
        else:
            left_id, left_frames, right_id, right_frames = id_b, frames_b, id_a, frames_a

        window = self.window_len
        crops_left = np.zeros((window, self.head_size, self.head_size, 3), dtype=np.uint8)
        crops_right = np.zeros((window, self.head_size, self.head_size, 3), dtype=np.uint8)
        maps = np.zeros((window, self.map_size[0], self.map_size[1], 3), dtype=np.uint8)

        for t in range(window):
            _, bbox_l, crop_l = left_frames[t]
            _, bbox_r, crop_r = right_frames[t]
            crops_left[t] = crop_l
            crops_right[t] = crop_r
            maps[t] = draw_pair_head_map(self.frame_shape, bbox_l, bbox_r, self.map_size)

        return PairSample(left_id, right_id, crops_left, crops_right, maps)
