"""Single-frame posture prediction with the exported model bundle.

Per frame: localise the person (detector -> pose box -> full frame), crop with the same
padding used in training, run MediaPipe pose on the crop, and combine

* the pose-feature classifier (only when a pose is found), and
* the image classifier (ONNX MobileNetV3 embedding + logistic regression)

by the late-fusion rule fixed during model selection. Scores are classifier
probabilities (not calibrated guarantees); see the model card.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from salat_vision.config import Config, get_config
from salat_vision.features.pose import PoseEstimator, PoseResult, crop_with_padding
from salat_vision.features.pose_features import compute_pose_features
from salat_vision.inference.bundle import ModelBundle
from salat_vision.inference.detector import PersonDetector
from salat_vision.labels import VISUAL_CLASSES, VISUAL_LABELS_ID
from salat_vision.models.cnn import IMAGENET_MEAN, IMAGENET_STD, letterbox
from salat_vision.models.rules import fuse


@dataclass
class FramePrediction:
    probs: np.ndarray | None
    probs_pose: np.ndarray | None
    probs_image: np.ndarray | None
    loc_source: str
    box: tuple[float, float, float, float] | None
    crop_box: tuple[int, int, int, int] | None
    pose: PoseResult | None
    image_size: tuple[int, int]
    timings_ms: dict = field(default_factory=dict)

    @property
    def top_class(self) -> str | None:
        return None if self.probs is None else VISUAL_CLASSES[int(np.argmax(self.probs))]

    @property
    def confidence(self) -> float | None:
        return None if self.probs is None else float(np.max(self.probs))

    def landmarks_image_norm(self) -> np.ndarray | None:
        """(33, 4) landmarks [x, y, visibility, presence] normalised to the *full* image."""
        if self.pose is None or not self.pose.detected or self.crop_box is None:
            return None
        X0, Y0, X1, Y1 = self.crop_box
        W, H = self.image_size
        lm = self.pose.landmarks
        x = (X0 + lm[:, 0] * (X1 - X0)) / W
        y = (Y0 + lm[:, 1] * (Y1 - Y0)) / H
        return np.stack([x, y, lm[:, 3], lm[:, 4]], axis=1)

    def to_dict(self, include_landmarks: bool = True) -> dict:
        W, H = self.image_size
        d = {
            "visual_class": self.top_class,
            "label": VISUAL_LABELS_ID.get(self.top_class) if self.top_class else None,
            "confidence": None if self.confidence is None else round(self.confidence, 4),
            "probabilities": None if self.probs is None else {c: round(float(p), 4) for c, p in zip(VISUAL_CLASSES, self.probs)},
            "probabilities_pose_model": None if self.probs_pose is None else {c: round(float(p), 4) for c, p in zip(VISUAL_CLASSES, self.probs_pose)},
            "probabilities_image_model": None if self.probs_image is None else {c: round(float(p), 4) for c, p in zip(VISUAL_CLASSES, self.probs_image)},
            "localization": self.loc_source,
            "person_box": None if self.box is None else [round(float(self.box[0]) / W, 4), round(float(self.box[1]) / H, 4),
                                                          round(float(self.box[2]) / W, 4), round(float(self.box[3]) / H, 4)],
            "pose_detected": bool(self.pose is not None and self.pose.detected),
            "timings_ms": {k: round(float(v), 2) for k, v in self.timings_ms.items()},
        }
        if include_landmarks:
            lm = self.landmarks_image_norm()
            d["landmarks"] = None if lm is None else np.round(lm, 4).tolist()
        return d


class PosturePredictor:
    """Not thread-safe (MediaPipe graphs): use one instance per worker thread."""

    def __init__(self, bundle: ModelBundle | None = None, cfg: Config | None = None):
        import joblib
        import onnxruntime as ort

        self.cfg = cfg or get_config()
        self.bundle = bundle or ModelBundle.load(cfg=self.cfg)
        m = self.bundle.manifest
        pipe = m["pipeline"]
        self.pad = float(pipe["crop_padding"])
        self.min_vis = float(pipe["min_landmark_visibility"])
        self.w_pose = float(pipe["fusion"]["w_pose"])
        tp = {e["role"]: e["name"] for e in m["third_party"]}
        pose_cfg = pipe["pose_estimator"]
        self.pose = PoseEstimator(self.bundle.pretrained(tp["pose_landmarker"]),
                                  min_detection_confidence=pose_cfg["min_detection_confidence"],
                                  min_presence_confidence=pose_cfg["min_presence_confidence"])
        det_cfg = pipe["person_detector"]
        self.detector = PersonDetector(self.bundle.pretrained(tp["person_detector"]),
                                       score_threshold=det_cfg["score_threshold"], max_results=det_cfg["max_results"])
        self.pose_clf = joblib.load(self.bundle.file(pipe["pose_classifier"])) if pipe.get("pose_classifier") else None
        img = pipe.get("image_model")
        self.img_clf, self.session, self.input_size = None, None, 224
        if img:
            so = ort.SessionOptions()
            so.intra_op_num_threads = int(img.get("threads", 2))
            self.session = ort.InferenceSession(str(self.bundle.file(img["embedder"])), sess_options=so,
                                                providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.input_size = int(img["input_size"])
            self.img_clf = joblib.load(self.bundle.file(img["classifier"]))

    # ------------------------------------------------------------------ components
    def localize(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        det = PersonDetector.primary(self.detector.detect(rgb), (w, h))
        if det is not None:
            return det.as_tuple(), "detector"
        full = self.pose.detect(rgb)
        b = full.bbox(pad=0.15) if full.detected else None
        if b is not None and (b[2] - b[0]) > 8 and (b[3] - b[1]) > 8:
            return b, "pose"
        return (0.0, 0.0, float(w), float(h)), "full_image"

    def embed(self, crop: np.ndarray) -> np.ndarray:
        sq = letterbox(crop, self.input_size).astype(np.float32) / 255.0
        x = ((sq - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None].astype(np.float32)
        return self.session.run(None, {self.input_name: x})[0]

    @staticmethod
    def _proba(clf, X: np.ndarray) -> np.ndarray:
        p = clf.predict_proba(X)[0]
        out = np.zeros(len(VISUAL_CLASSES))
        out[np.asarray(clf.classes_, dtype=int)] = p
        return out

    # ------------------------------------------------------------------ main entry
    def predict(self, rgb: np.ndarray) -> FramePrediction:
        t = {}
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        box, source = self.localize(rgb)
        t["localize"] = (time.perf_counter() - t0) * 1000
        crop, crop_box = crop_with_padding(rgb, box, self.pad)
        t1 = time.perf_counter()
        pose = self.pose.detect(crop)
        t["pose"] = (time.perf_counter() - t1) * 1000
        p_pose = p_img = None
        t2 = time.perf_counter()
        # With an image model, frames without a pose are left to it (as evaluated in CV);
        # a pose-only pipeline classifies imputed features, also as evaluated.
        if self.pose_clf is not None and (pose.detected or self.session is None):
            feats = compute_pose_features(pose, (box[2] - box[0], box[3] - box[1]), min_visibility=self.min_vis)
            p_pose = self._proba(self.pose_clf, feats[None])
        t["pose_classifier"] = (time.perf_counter() - t2) * 1000
        t3 = time.perf_counter()
        if self.session is not None:
            p_img = self._proba(self.img_clf, self.embed(crop))
        t["image_model"] = (time.perf_counter() - t3) * 1000
        if p_img is not None and p_pose is not None:
            probs = fuse(p_pose[None], p_img[None], w_pose=self.w_pose)[0]
        else:
            probs = p_img if p_img is not None else p_pose
        t["total"] = (time.perf_counter() - t0) * 1000
        return FramePrediction(probs, p_pose, p_img, source, box, crop_box, pose, (w, h), t)

    def close(self) -> None:
        self.pose.close()
        self.detector.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
