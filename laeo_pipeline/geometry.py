"""
Bounding-box and head-map geometry helpers.

The head-map drawing code (`draw_pair_head_map` and helpers) is a direct port of
`mj_drawHeadMapFromBBs` / `mj_cropImageFromBBs` / `mj_gkern` from the original
LAEO-Net++ codebase (utils/ln_laeoImage.py), rewritten without any TensorFlow/Keras
dependency so it can be reused by the PyTorch pipeline. The pretrained LAEO-Net++
weights were trained against maps built by that exact algorithm, so it is kept
faithful to the original rather than reinvented.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence, Tuple

import cv2
import numpy as np

BBox = Tuple[float, float, float, float]


def clip_bbox(box: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = box
    x1 = min(max(x1, 0.0), width - 1)
    y1 = min(max(y1, 0.0), height - 1)
    x2 = min(max(x2, 0.0), width - 1)
    y2 = min(max(y2, 0.0), height - 1)
    return x1, y1, x2, y2


def expand_face_to_head(box: BBox, frame_shape: Sequence[int], scale: float = 1.8) -> BBox:
    """Expand a face bounding box into an approximate head bounding box.

    The box is scaled by `scale` around its own center and clipped to the frame.
    """
    height, width = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale

    nx1 = cx - w / 2.0
    ny1 = cy - h / 2.0
    nx2 = cx + w / 2.0
    ny2 = cy + h / 2.0

    return clip_bbox((nx1, ny1, nx2, ny2), width, height)


def to_squared(box: BBox, frame_shape: Sequence[int], margin: float = 0.05) -> BBox:
    """Convert an arbitrary box into a square box (centered, with a small margin).

    Port of `mj_detBB2squared`, adapted to 0-indexed clipping.
    """
    height, width = frame_shape[0], frame_shape[1]
    x1, y1, x2, y2 = box

    w = (x2 - x1) + 1.0
    h = (y2 - y1) + 1.0
    side = max(w, h) * (1.0 + margin)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    nx1 = cx - side / 2.0
    ny1 = cy - side / 2.0
    nx2 = cx + side / 2.0
    ny2 = cy + side / 2.0

    return clip_bbox((nx1, ny1, nx2, ny2), width, height)


def iou(box_a: BBox, box_b: BBox) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa + 1.0)
    inter_h = max(0.0, yb - ya + 1.0)
    inter = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0] + 1.0) * max(0.0, box_a[3] - box_a[1] + 1.0)
    area_b = max(0.0, box_b[2] - box_b[0] + 1.0) * max(0.0, box_b[3] - box_b[1] + 1.0)

    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def crop_head(frame: np.ndarray, box: BBox, out_size: int = 64) -> np.ndarray:
    """Crop a squared head region from `frame` and resize it to (out_size, out_size)."""
    sq = to_squared(box, frame.shape)
    x1, y1, x2, y2 = (int(round(v)) for v in sq)
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((out_size, out_size, 3), dtype=frame.dtype)
    return cv2.resize(crop, (out_size, out_size))


@lru_cache(maxsize=512)
def _gaussian_blob(tam: int) -> np.ndarray:
    """A (tam, tam) Gaussian blob, rescaled to [0, 255]. Cached by size since the same
    head/box size recurs across frames and across pairs sharing a track -- without
    caching, a crowded scene with many pairs re-runs the same Gaussian filter
    thousands of times per video frame.
    """
    from scipy.ndimage import gaussian_filter

    sigma = max(round(tam / 4), 1e-3)
    inp = np.zeros((tam, tam), dtype=np.float64)
    inp[tam // 2, tam // 2] = 1.0
    kernel = gaussian_filter(inp, sigma)

    kernel = kernel - kernel.min()
    kmax = kernel.max()
    if kmax > 0:
        kernel = 255.0 * kernel / kmax
    return kernel


def _bbs_extent(boxes: Sequence[BBox]) -> Tuple[int, int, int, int]:
    xs = [c for b in boxes for c in (b[0], b[2])]
    ys = [c for b in boxes for c in (b[1], b[3])]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _crop_to_extent_and_square(canvas: np.ndarray, boxes: Sequence[BBox]) -> np.ndarray:
    """Crop `canvas` horizontally to the extent of `boxes` (+-20px), full height kept,
    then letterbox-pad to a square. Port of `mj_cropImageFromBBs`.
    """
    min_x, _min_y, max_x, _max_y = _bbs_extent(boxes)

    crop = canvas[0:canvas.shape[0], max(min_x - 20, 0):min(max_x + 20, canvas.shape[1])]
    crop_h, crop_w = crop.shape[0], crop.shape[1]
    side = max(crop_h, crop_w)

    square = np.zeros((side, side, 3), dtype=np.uint8)
    y0 = round(side / 2) - round(crop_h / 2)
    x0 = round(side / 2) - round(crop_w / 2)
    square[y0:y0 + crop_h, x0:x0 + crop_w] = crop

    return square


def draw_pair_head_map(
    frame_shape: Sequence[int],
    box_left: BBox,
    box_right: BBox,
    target_size: Tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Render the two-head spatial map used by the LAEO-Net++ map branch.

    Channel 0 encodes `box_left`, channel 1 encodes `box_right` (as a small Gaussian
    blob sized after each head), channel 2 is left empty. The canvas is then cropped
    to the horizontal extent of the two heads, letterboxed to a square and resized to
    `target_size`. Faithful port of `mj_drawHeadMapFromBBs`.
    """
    height, width = frame_shape[0], frame_shape[1]
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    tam_limit = min(height, width)

    boxes = (box_left, box_right)
    for channel, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        w = (x2 - x1) + 1.0
        h = (y2 - y1) + 1.0
        tam = int(max(3, max(w, h)))
        if tam < 3 or tam > tam_limit:
            continue

        kernel = _gaussian_blob(tam)

        j_start = int(max(0, x1))
        i_start = int(max(0, y1))
        j_end = int(x1) + tam
        i_end = int(y1) + tam
        j_over = max(0, j_end - canvas.shape[1])
        i_over = max(0, i_end - canvas.shape[0])
        j0 = int(abs(min(0, x1)))
        i0 = int(abs(min(0, y1)))

        dst_j_end = j_end - j_over
        dst_i_end = i_end - i_over
        if dst_j_end <= j_start or dst_i_end <= i_start:
            continue

        patch = kernel[i0:kernel.shape[0] - i_over, j0:kernel.shape[1] - j_over]
        canvas[i_start:dst_i_end, j_start:dst_j_end, channel] = patch.astype(np.uint8)

    squared = _crop_to_extent_and_square(canvas, boxes)
    return cv2.resize(squared, target_size)
