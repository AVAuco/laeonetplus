"""End-to-end LAEO-Net++ pipeline: head detection -> tracking -> LAEO scoring -> output."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .deep_sort.detection import Detection as DsDetection
from .deep_sort.tracker import Tracker
from .detection import HeadDetector
from .geometry import clip_bbox
from .laeo_infer import LaeoScorer
from .pair_buffer import PairBuffer
from .reid_embedder import ReIDEmbedder
from .visualization import draw_frame_index, draw_heads, draw_pair


def _bbox_to_tlwh(bbox) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)


class LaeoPipeline:
    def __init__(
        self,
        face_weights: str = "yolov8n-face.pt",
        laeo_json: str = "models_pt/laeonet_ava.json",
        laeo_weights: str = "models_pt/laeonet_ava_weights.h5",
        mean_map_path: str = "models/meanmaps10.npy",
        device: str = "cpu",
        detector_conf: float = 0.35,
        head_scale: float = 1.8,
        window_len: int = 10,
        max_age: int = 30,
        n_init: int = 3,
    ) -> None:
        self.detector = HeadDetector(weights=face_weights, conf=detector_conf, device=device, head_scale=head_scale)
        self.embedder = ReIDEmbedder(device=device)
        self.tracker = Tracker(max_age=max_age, n_init=n_init)
        self.scorer = LaeoScorer(laeo_json, laeo_weights, mean_map_path, device=device)
        self.window_len = window_len

    def run(
        self,
        video_path: str,
        output_video: Optional[str] = None,
        output_json: Optional[str] = None,
        threshold: float = 0.5,
        max_frames: Optional[int] = None,
    ) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Could not open output video for writing: {output_video}")

        pair_buffer = PairBuffer((height, width), window_len=self.window_len)
        pair_records: Dict[Tuple[int, int], List[dict]] = {}
        known_track_ids: set = set()

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break

                detections = self.detector.detect(frame)
                crops = [
                    frame[
                        max(0, int(d.bbox[1])):max(1, int(d.bbox[3])),
                        max(0, int(d.bbox[0])):max(1, int(d.bbox[2])),
                    ]
                    for d in detections
                ]
                crops = [c if c.size else np.zeros((64, 64, 3), dtype=np.uint8) for c in crops]
                features = self.embedder.embed(crops)

                ds_detections = [
                    DsDetection(_bbox_to_tlwh(d.bbox), d.score, features[i])
                    for i, d in enumerate(detections)
                ]

                self.tracker.predict()
                self.tracker.update(ds_detections)

                current_ids = {t.track_id for t in self.tracker.tracks}
                for stale_id in known_track_ids - current_ids:
                    pair_buffer.remove_track(stale_id)
                known_track_ids = current_ids

                current_tracks: Dict[int, Tuple[float, float, float, float]] = {}
                for track in self.tracker.tracks:
                    if not track.is_confirmed():
                        continue
                    bbox = clip_bbox(tuple(track.to_tlbr().tolist()), width, height)
                    current_tracks[track.track_id] = bbox

                ready_pairs = pair_buffer.update(frame_idx, frame, list(current_tracks.items()))

                frame_pair_probs = []
                for sample in ready_pairs:
                    probability = self.scorer.score(sample)
                    key = sample.pair_key
                    pair_records.setdefault(key, []).append({"frame": frame_idx, "probability": probability})
                    frame_pair_probs.append((sample.track_a, sample.track_b, probability))

                if writer is not None:
                    draw_heads(frame, current_tracks)
                    for track_a, track_b, probability in frame_pair_probs:
                        draw_pair(frame, current_tracks[track_a], current_tracks[track_b], probability, threshold)
                    draw_frame_index(frame, frame_idx)
                    writer.write(frame)

                frame_idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        result = {
            "video": video_path,
            "fps": fps,
            "width": width,
            "height": height,
            "num_frames": frame_idx,
            "threshold": threshold,
            "window_len": self.window_len,
            "pairs": [
                {"track1": key[0], "track2": key[1], "frames": frames}
                for key, frames in sorted(pair_records.items())
            ],
        }

        if output_json:
            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)

        return result
