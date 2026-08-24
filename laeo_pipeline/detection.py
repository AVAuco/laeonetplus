"""Head detection: YOLOv8-face (ultralytics) with face -> head bbox expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .geometry import BBox, expand_face_to_head


@dataclass
class HeadDetection:
    bbox: BBox  # (x1, y1, x2, y2), already expanded to head size
    score: float


class HeadDetector:
    """Wraps an Ultralytics YOLOv8 face-detection checkpoint and expands each
    detected face box into an approximate head box.
    """

    def __init__(
        self,
        weights: str = "yolov8n-face.pt",
        conf: float = 0.35,
        iou: float = 0.5,
        device: Optional[str] = None,
        head_scale: float = 1.8,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "The 'ultralytics' package is required for face detection. "
                "Install it with `pip install ultralytics`."
            ) from exc

        try:
            self.model = YOLO(weights)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load YOLOv8-face weights from '{weights}'. Download a "
                "YOLOv8-face checkpoint (a YOLOv8 model trained on face detection, "
                "e.g. from the yolov8-face project) and pass its path via "
                "--face_model."
            ) from exc

        self.conf = conf
        self.iou = iou
        self.device = device
        self.head_scale = head_scale

    def detect(self, frame: np.ndarray) -> List[HeadDetection]:
        results = self.model.predict(
            frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False
        )[0]

        detections: List[HeadDetection] = []
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            score = float(box.conf[0])
            head_box = expand_face_to_head(tuple(xyxy), frame.shape, self.head_scale)
            detections.append(HeadDetection(bbox=head_box, score=score))

        return detections
