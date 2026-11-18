"""MediaPipe Pose Landmarker (Tasks API) wrapper.

Returns 33 BlazePose landmarks per detected person: normalised image coordinates
(x, y in [0, 1], z, visibility, presence) and metric *world* coordinates (metres,
origin between the hips). Only the single most prominent person is used.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# Quieten MediaPipe / LiteRT native logging unless the user configured it explicitly.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]
LM = {name: i for i, name in enumerate(LANDMARK_NAMES)}

#: Skeleton edges used for drawing.
POSE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
    (24, 26), (26, 28), (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32), (0, 11), (0, 12),
]


@dataclass
class PoseResult:
    detected: bool
    image_size: tuple[int, int]  # (width, height) of the image passed to the estimator
    landmarks: np.ndarray | None = None  # (33, 5): x, y, z, visibility, presence (normalised)
    world: np.ndarray | None = None  # (33, 3) metres

    def pixel_xy(self) -> np.ndarray | None:
        if self.landmarks is None:
            return None
        w, h = self.image_size
        return self.landmarks[:, :2] * np.array([w, h], dtype=np.float32)

    def bbox(self, min_visibility: float = 0.3, pad: float = 0.1) -> tuple[float, float, float, float] | None:
        """Pixel bounding box (x0, y0, x1, y1) around visible landmarks, padded and clipped."""
        xy = self.pixel_xy()
        if xy is None:
            return None
        vis = self.landmarks[:, 3] >= min_visibility
        if vis.sum() < 4:
            vis = np.ones(len(xy), dtype=bool)
        pts = xy[vis]
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        w, h = self.image_size
        dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
        return (max(0.0, x0 - dx), max(0.0, y0 - dy), min(float(w), x1 + dx), min(float(h), y1 + dy))

    def to_dict(self) -> dict:
        return {
            "detected": self.detected,
            "image_size": list(self.image_size),
            "landmarks": None if self.landmarks is None else np.round(self.landmarks, 5).tolist(),
            "world": None if self.world is None else np.round(self.world, 5).tolist(),
        }


class PoseEstimator:
    """Thin wrapper around ``mediapipe.tasks.python.vision.PoseLandmarker``.

    ``mode='image'`` treats frames independently; ``mode='video'`` enables MediaPipe's
    internal tracking and requires monotonically increasing timestamps.
    """

    def __init__(self, model_path: str | Path, *, mode: str = "image", min_detection_confidence: float = 0.3,
                 min_presence_confidence: float = 0.3, min_tracking_confidence: float = 0.3,
                 min_input_side: int = 256):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.mode = mode
        self.min_input_side = min_input_side
        running_mode = vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self.model_name = Path(model_path).stem

    def detect(self, rgb: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        """Run pose estimation on an RGB ``uint8`` image (any size)."""
        h, w = rgb.shape[:2]
        img = rgb
        if min(h, w) < self.min_input_side:  # tiny crops: upscale (coordinates are normalised anyway)
            s = self.min_input_side / max(min(h, w), 1)
            img = cv2.resize(rgb, (max(1, int(round(w * s))), max(1, int(round(h * s)))), interpolation=cv2.INTER_CUBIC)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
        if self.mode == "video":
            res = self._landmarker.detect_for_video(mp_image, int(timestamp_ms or 0))
        else:
            res = self._landmarker.detect(mp_image)
        if not res.pose_landmarks:
            return PoseResult(False, (w, h))
        lms = res.pose_landmarks[0]
        arr = np.array([[p.x, p.y, p.z, p.visibility if p.visibility is not None else 0.0,
                         p.presence if p.presence is not None else 0.0] for p in lms], dtype=np.float32)
        world = None
        if res.pose_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], dtype=np.float32)
        return PoseResult(True, (w, h), arr, world)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def crop_with_padding(rgb: np.ndarray, box: tuple[float, float, float, float], pad: float) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop ``box`` (x0, y0, x1, y1) enlarged by ``pad`` x its size on each side, clipped to the image."""
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    X0 = int(max(0, np.floor(x0 - pad * bw)))
    Y0 = int(max(0, np.floor(y0 - pad * bh)))
    X1 = int(min(w, np.ceil(x1 + pad * bw)))
    Y1 = int(min(h, np.ceil(y1 + pad * bh)))
    return rgb[Y0:Y1, X0:X1], (X0, Y0, X1, Y1)
