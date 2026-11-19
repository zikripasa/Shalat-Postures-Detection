"""Model service: loads the bundle once, hands out one predictor per worker thread."""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from pathlib import Path

from salat_vision.config import get_config
from salat_vision.inference.bundle import BundleError, ModelBundle
from salat_vision.inference.predictor import PosturePredictor
from salat_vision.labels import (PHASE_EVIDENCE, PHASE_LABELS_EN, PHASE_LABELS_ID, PHASE_TO_VISUAL, PHASES,
                                 VISUAL_CLASSES, VISUAL_LABELS_EN, VISUAL_LABELS_ID)

from .settings import Settings

log = logging.getLogger("salat_vision.api")


class ModelUnavailable(RuntimeError):
    pass


class ModelService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = get_config()
        self.bundle: ModelBundle | None = None
        self.error: str | None = "model not loaded yet"
        self.loaded_at: float | None = None
        self.started_at = time.time()
        self._local = threading.local()
        self._predictors: list[PosturePredictor] = []
        self._plock = threading.Lock()
        self._image_lock = threading.Lock()
        self._image_predictor: PosturePredictor | None = None

    # ------------------------------------------------------------------ lifecycle
    def load(self) -> None:
        try:
            self.bundle = ModelBundle.load(self.settings.model_dir, cfg=self.cfg, verify=True)
            # warm-up: build one predictor so failures surface at start-up, not on first request
            self._image_predictor = self._new_predictor()
            self.error = None
            self.loaded_at = time.time()
            log.info("model bundle %s loaded", self.bundle.version)
        except (BundleError, FileNotFoundError, OSError, RuntimeError, KeyError) as exc:
            self.bundle = None
            self.error = str(exc)
            log.warning("model unavailable: %s", exc)

    def _new_predictor(self) -> PosturePredictor:
        p = PosturePredictor(self.bundle, cfg=self.cfg)
        with self._plock:
            self._predictors.append(p)
        return p

    def close(self) -> None:
        with self._plock:
            for p in self._predictors:
                try:
                    p.close()
                except Exception:  # pragma: no cover
                    pass
            self._predictors.clear()

    @property
    def available(self) -> bool:
        return self.bundle is not None and self.error is None

    def require(self) -> None:
        if not self.available:
            raise ModelUnavailable(self.error or "model unavailable")

    # ------------------------------------------------------------------ inference
    def predictor_for_thread(self) -> PosturePredictor:
        self.require()
        p = getattr(self._local, "predictor", None)
        if p is None:
            p = self._new_predictor()
            self._local.predictor = p
        return p

    def predict_image(self, rgb):
        self.require()
        with self._image_lock:
            return self._image_predictor.predict(rgb)

    # ------------------------------------------------------------------ info
    def _artifact(self, *parts: str) -> Path:
        return self.cfg.path("artifacts_dir").joinpath(*parts)

    def _read_json(self, *parts: str):
        p = self._artifact(*parts)
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None

    def _read_csv(self, *parts: str):
        p = self._artifact(*parts)
        if not p.is_file():
            return None
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k, v in r.items():
                try:
                    r[k] = float(v) if v not in ("", None) and k not in ("model", "family") else (v or None)
                except ValueError:
                    pass
        return rows

    def model_info(self) -> dict:
        classes = [{"id": c, "label": VISUAL_LABELS_ID[c], "label_en": VISUAL_LABELS_EN[c]} for c in VISUAL_CLASSES]
        phases = [{"id": p, "label": PHASE_LABELS_ID[p], "label_en": PHASE_LABELS_EN[p], "visual_class": PHASE_TO_VISUAL[p],
                   "evidence": PHASE_EVIDENCE[p]} for p in PHASES]
        info = {
            "available": self.available,
            "reason": None if self.available else self.error,
            "classes": classes,
            "phases": phases,
            "model_comparison": self._read_csv("metrics", "model_comparison.csv"),
            "temporal_benchmark": self._read_json("metrics", "temporal_benchmark.json"),
            "runtime_benchmark": self._read_json("metrics", "runtime_benchmark.json"),
            "dataset": None,
            "demo_video_available": self.demo_video_path() is not None,
        }
        audit = self._read_json("metrics", "dataset_audit.json")
        if audit:
            info["dataset"] = {k: audit.get(k) for k in ("n_image_files", "n_labeled_images", "n_labeled_instances",
                                                         "instances_per_class", "independent_groups_per_class",
                                                         "n_images_without_annotation", "n_orphan_annotations",
                                                         "phase_support", "provenance")}
            info["dataset"]["n_groups"] = audit.get("leakage", {}).get("n_groups_labeled")
        if self.bundle is not None:
            m = self.bundle.manifest
            info.update({
                "version": m.get("version"),
                "created_utc": m.get("created_utc"),
                "pipeline": m.get("pipeline"),
                "training": m.get("training"),
                "cv_metrics": m.get("cv_metrics"),
                "intended_use": m.get("intended_use"),
                "out_of_scope": m.get("out_of_scope"),
                "limitations": m.get("limitations"),
                "third_party": [{k: e[k] for k in ("role", "name", "license")} for e in m.get("third_party", [])],
                "environment": m.get("environment"),
            })
        return info

    def demo_video_path(self) -> Path | None:
        p = self._artifact("demo", "demo_sequence.mp4")
        return p if p.is_file() else None
