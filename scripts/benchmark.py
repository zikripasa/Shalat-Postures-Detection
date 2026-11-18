#!/usr/bin/env python
"""Measure inference latency on this machine (CPU) and pose-detector variants.

Writes
  artifacts/metrics/runtime_benchmark.json        per-stage latency of the deployed pipeline
  artifacts/metrics/pose_variant_benchmark.json   detection rate + latency of lite/full/heavy pose models
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from salat_vision.config import get_config  # noqa: E402
from salat_vision.data.dataset import build_index  # noqa: E402
from salat_vision.data.voc import load_image_rgb  # noqa: E402
from salat_vision.features.pose import PoseEstimator, crop_with_padding  # noqa: E402
from salat_vision.inference.predictor import PosturePredictor  # noqa: E402
from salat_vision.labels import VISUAL_CLASSES  # noqa: E402


def stats(ms: list[float]) -> dict:
    ms = sorted(ms)
    return {"n": len(ms), "mean_ms": round(statistics.fmean(ms), 2), "median_ms": round(statistics.median(ms), 2),
            "p95_ms": round(ms[int(0.95 * (len(ms) - 1))], 2), "min_ms": round(ms[0], 2)}


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main() -> int:
    cfg = get_config()
    idx = build_index(cfg=cfg)
    li = idx.labeled_instances
    images = [load_image_rgb(idx.dataset_dir / f) for f in li.drop_duplicates("image_id")["file"]]
    # analysis frames are capped at 1280 px (as in video analysis)
    import cv2

    def cap(img, side=1280):
        s = min(1.0, side / max(img.shape[:2]))
        return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))) if s < 1 else img

    frames = [cap(im) for im in images]
    hw = {"cpu": cpu_name(), "logical_cores": os.cpu_count(), "python": platform.python_version(),
          "platform": platform.platform()}

    # ---- deployed pipeline, per stage --------------------------------------------------
    pred = PosturePredictor(cfg=cfg)
    for f in frames[:3]:
        pred.predict(f)  # warm-up
    stage: dict[str, list[float]] = {}
    for _ in range(3):
        for f in frames:
            r = pred.predict(f)
            for k, v in r.timings_ms.items():
                stage.setdefault(k, []).append(v)
    crops = [crop_with_padding(im, (row.xmin, row.ymin, row.xmax, row.ymax), 0.15)[0]
             for im, row in zip([load_image_rgb(idx.dataset_dir / f) for f in li["file"]], li.itertuples())]
    ort_ms = []
    for c in crops * 3:
        t = time.perf_counter()
        pred.embed(c)
        ort_ms.append((time.perf_counter() - t) * 1000)
    pred.close()

    import torch

    from salat_vision.models.cnn import build_backbone, preprocess_batch

    torch.set_num_threads(os.cpu_count() or 1)
    body, _ = build_backbone("mobilenet_v3_large", cfg=cfg)
    body.eval()
    torch_ms = []
    with torch.inference_mode():
        for c in crops[:3]:
            body(torch.from_numpy(preprocess_batch([c], 224)))
        for c in crops * 3:
            t = time.perf_counter()
            body(torch.from_numpy(preprocess_batch([c], 224)))
            torch_ms.append((time.perf_counter() - t) * 1000)

    total = stats(stage["total"])
    runtime = {
        "hardware": hw,
        "input": f"{len(frames)} dataset images, longest side capped at 1280 px, 3 passes after warm-up",
        "stages": {k: stats(v) for k, v in stage.items()},
        "embedder": {"onnxruntime_ms": stats(ort_ms), "pytorch_ms": stats(torch_ms)},
        "throughput_fps": round(1000.0 / total["mean_ms"], 2),
        "note": "Single-process CPU inference, no GPU. The video pipeline analyses frames sampled at inference.sample_fps.",
    }

    # ---- pose landmarker variants --------------------------------------------------------
    pre = cfg.path("pretrained_dir")
    variants = {}
    for v in ("lite", "full", "heavy"):
        det = {c: [] for c in VISUAL_CLASSES}
        ms = []
        with PoseEstimator(pre / f"pose_landmarker_{v}.task") as pe:
            pe.detect(crops[0])
            for c, lab in zip(crops, li["label"]):
                t = time.perf_counter()
                res = pe.detect(c)
                ms.append((time.perf_counter() - t) * 1000)
                det[lab].append(res.detected)
        variants[v] = {"latency": stats(ms), "detection_rate": round(float(np.mean(sum(det.values(), []))), 4),
                       "detection_rate_by_class": {c: round(float(np.mean(d)), 4) for c, d in det.items()},
                       "size_mb": round((pre / f"pose_landmarker_{v}.task").stat().st_size / 1e6, 1)}
    pose_bench = {"hardware": hw, "input": f"{len(crops)} annotated person crops (15% padding)", "variants": variants,
                  "selected": cfg["models_remote"]["pose_landmarker"]["file"]}

    out = cfg.path("artifacts_dir") / "metrics"
    (out / "runtime_benchmark.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (out / "pose_variant_benchmark.json").write_text(json.dumps(pose_bench, indent=2), encoding="utf-8")
    print(json.dumps({"stages_mean_ms": {k: v["mean_ms"] for k, v in runtime["stages"].items()},
                      "fps": runtime["throughput_fps"], "ort": runtime["embedder"]["onnxruntime_ms"]["mean_ms"],
                      "torch": runtime["embedder"]["pytorch_ms"]["mean_ms"],
                      "pose": {k: (v["detection_rate"], v["latency"]["mean_ms"]) for k, v in variants.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
