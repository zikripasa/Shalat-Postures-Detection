"""Person localisation: MediaPipe Object Detector (EfficientDet-Lite0, COCO 'person')."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("GLOG_minloglevel", "2")


@dataclass
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    score: float

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class PersonDetector:
    def __init__(self, model_path: str | Path, *, score_threshold: float = 0.3, max_results: int = 5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.ObjectDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            max_results=max_results,
            score_threshold=score_threshold,
            category_allowlist=["person"],
        )
        self._det = vision.ObjectDetector.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h, w = rgb.shape[:2]
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._det.detect(mp_image)
        out = []
        for d in res.detections:
            bb = d.bounding_box
            score = float(d.categories[0].score) if d.categories else 0.0
            x0, y0 = max(0.0, float(bb.origin_x)), max(0.0, float(bb.origin_y))
            x1, y1 = min(float(w), x0 + float(bb.width)), min(float(h), y0 + float(bb.height))
            if x1 - x0 > 2 and y1 - y0 > 2:
                out.append(Detection(x0, y0, x1, y1, score))
        return out

    @staticmethod
    def primary(dets: list[Detection], image_wh: tuple[int, int]) -> Detection | None:
        """The most prominent person: confidence x sqrt(relative area)."""
        if not dets:
            return None
        W, H = image_wh
        return max(dets, key=lambda d: d.score * np.sqrt(d.area / float(W * H)))

    def close(self) -> None:
        self._det.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
