#!/usr/bin/env python
"""Download third-party pretrained models into ``models/pretrained``.

* MediaPipe Pose Landmarker (lite / full / heavy) — Apache-2.0
* MediaPipe Object Detector EfficientDet-Lite0 (COCO, used for the 'person' class) — Apache-2.0
* torchvision ImageNet weights for MobileNetV3-Large and ResNet18 — BSD-3-Clause
  (stored under ``models/pretrained/torch`` via ``TORCH_HOME``)

Records SHA-256 checksums in ``models/pretrained/manifest.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salat_vision.config import get_config  # noqa: E402

MEDIAPIPE_MODELS = {
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/latest/efficientdet_lite0.tflite",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-torch", action="store_true", help="Do not fetch torchvision weights")
    args = ap.parse_args()
    cfg = get_config()
    out = cfg.path("pretrained_dir")
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for name, url in MEDIAPIPE_MODELS.items():
        dest = out / name
        if not dest.exists():
            print(f"Downloading {name} ...")
            download(url, dest)
        manifest[name] = {"url": url, "bytes": dest.stat().st_size, "sha256": sha256(dest), "license": "Apache-2.0"}
        print(f"  {name}: {dest.stat().st_size/1e6:.1f} MB")

    if not args.skip_torch:
        os.environ["TORCH_HOME"] = str(out / "torch")
        import torchvision.models as tvm

        for builder, weights in (("mobilenet_v3_large", tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
                                 ("resnet18", tvm.ResNet18_Weights.IMAGENET1K_V1)):
            getattr(tvm, builder)(weights=weights)
            fname = Path(weights.url).name
            path = out / "torch" / "hub" / "checkpoints" / fname
            manifest[f"torchvision/{builder}"] = {"url": weights.url, "weights": str(weights), "bytes": path.stat().st_size,
                                                   "sha256": sha256(path), "license": "BSD-3-Clause (torchvision)"}
            print(f"  torchvision {builder}: {path.stat().st_size/1e6:.1f} MB")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
