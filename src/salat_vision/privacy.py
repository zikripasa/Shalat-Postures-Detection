"""Face anonymisation for every image of a person that leaves the pipeline as an artefact
(sample grids, demo video, screenshots). Uses two independent cues and blurs their union:

* head landmarks (nose, eyes, ears, mouth) from MediaPipe Pose on each person crop, and
* MediaPipe BlazeFace detections on the whole image.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from salat_vision.features.pose import PoseEstimator, PoseResult, crop_with_padding

os.environ.setdefault("GLOG_minloglevel", "2")
HEAD_LANDMARKS = list(range(11))


class FaceDetector:
    def __init__(self, model_path: str | Path, min_confidence: float = 0.6):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        opts = vision.FaceDetectorOptions(base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
                                          min_detection_confidence=min_confidence)
        self._det = vision.FaceDetector.create_from_options(opts)

    def detect(self, rgb: np.ndarray) -> list[tuple[float, float, float, float]]:
        res = self._det.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        out = []
        for d in res.detections:
            b = d.bounding_box
            out.append((b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height))
        return out

    def close(self) -> None:
        self._det.close()


def head_box_from_pose(pose: PoseResult, crop_box: tuple[int, int, int, int], min_vis: float = 0.3,
                       max_size: float | None = None):
    """Head bounding box in image pixels from face landmarks (None if not visible)."""
    if pose is None or not pose.detected:
        return None
    X0, Y0, X1, Y1 = crop_box
    lm = pose.landmarks
    vis = lm[HEAD_LANDMARKS, 3] >= min_vis
    if vis.sum() < 2:
        return None
    xs = X0 + lm[HEAD_LANDMARKS, 0][vis] * (X1 - X0)
    ys = Y0 + lm[HEAD_LANDMARKS, 1][vis] * (Y1 - Y0)
    # head size from shoulder width when available (face landmarks span only the face)
    sh = lm[[11, 12]]
    shoulder_w = abs(sh[0, 0] - sh[1, 0]) * (X1 - X0) if (sh[:, 3] >= min_vis).all() else 0.0
    span = max(xs.max() - xs.min(), ys.max() - ys.min())
    size = max(span * 2.4, shoulder_w * 0.75, 0.12 * max(X1 - X0, Y1 - Y0))
    if max_size is not None:
        size = min(size, max_size)
    cx, cy = xs.mean(), ys.mean()
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


def blur_regions(rgb: np.ndarray, boxes, strength: float = 0.18) -> np.ndarray:
    out = rgb.copy()
    h, w = out.shape[:2]
    for x0, y0, x1, y1 in boxes:
        x0, y0 = int(max(0, x0)), int(max(0, y0))
        x1, y1 = int(min(w, x1)), int(min(h, y1))
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        roi = out[y0:y1, x0:x1]
        # pixelate then blur: irreversible at this scale, still reads as "a head"
        small = cv2.resize(roi, (max(1, (x1 - x0) // 12), max(1, (y1 - y0) // 12)), interpolation=cv2.INTER_AREA)
        roi = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        k = max(3, int(max(x1 - x0, y1 - y0) * strength) | 1)
        out[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)
    return out


class Anonymizer:
    def __init__(self, pretrained_dir: Path, pose_model: str = "pose_landmarker_full.task"):
        self.pose = PoseEstimator(pretrained_dir / pose_model)
        face_path = pretrained_dir / "blaze_face_short_range.tflite"
        self.face = FaceDetector(face_path) if face_path.is_file() else None

    def __call__(self, rgb: np.ndarray, person_boxes: list[tuple[float, float, float, float]] | None = None,
                 pad: float = 0.15) -> tuple[np.ndarray, list]:
        h, w = rgb.shape[:2]
        boxes = person_boxes or [(0.0, 0.0, float(w), float(h))]
        regions = []
        for b in boxes:
            crop, cbox = crop_with_padding(rgb, b, pad)
            # a head is never larger than ~40% of the person's longest side
            hb = head_box_from_pose(self.pose.detect(crop), cbox, max_size=0.4 * max(b[2] - b[0], b[3] - b[1]))
            if hb:
                regions.append(hb)
        if self.face is not None:
            for x0, y0, x1, y1 in self.face.detect(rgb):
                cx, cy, size = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
                # keep faces that sit inside a person box and are plausibly sized for it
                if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] and size <= 0.5 * max(b[2] - b[0], b[3] - b[1])
                       for b in boxes):
                    m = 0.35 * size
                    regions.append((x0 - m, y0 - m, x1 + m, y1 + m))
        return blur_regions(rgb, regions), regions

    def close(self) -> None:
        self.pose.close()
        if self.face is not None:
            self.face.close()
