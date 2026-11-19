#!/usr/bin/env python
"""Download third-party pretrained models into ``models/pretrained``.

* MediaPipe Pose Landmarker (lite / full / heavy) — Apache-2.0
* MediaPipe Object Detector EfficientDet-Lite0 (COCO, used for the 'person' class) — Apache-2.0
* MediaPipe BlazeFace (short range) — Apache-2.0, used only to blur faces in generated artefacts
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

# Version-pinned URLs. The ".../latest/..." aliases can change without notice, and the model
# bundle verifies these files by SHA-256 (models/<bundle>/manifest.json, "third_party").
_MP = "https://storage.googleapis.com/mediapipe-models"
MEDIAPIPE_MODELS = {
    "pose_landmarker_lite.task": f"{_MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{_MP}/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "pose_landmarker_heavy.task": f"{_MP}/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
    "efficientdet_lite0.tflite": f"{_MP}/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite",
    # only used to blur faces in generated figures / demo video (privacy), not for recognition
    "blaze_face_short_range.tflite": f"{_MP}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
}


def expected_checksums(cfg) -> dict[str, str]:
    """SHA-256 that the exported model bundle expects for each third-party file ({} if no bundle)."""
    mf = cfg.path("models_dir") / cfg["final_model"]["bundle_name"] / "manifest.json"
    if not mf.is_file():
        return {}
    return {e["name"]: e["sha256"] for e in json.loads(mf.read_text()).get("third_party", []) if e.get("sha256")}


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
    mpath = out / "manifest.json"
    # merge with earlier runs so e.g. --skip-torch keeps previously recorded torchvision entries
    manifest: dict[str, dict] = json.loads(mpath.read_text()) if mpath.is_file() else {}
    expected = expected_checksums(cfg)
    mismatched: list[str] = []
    for name, url in MEDIAPIPE_MODELS.items():
        dest = out / name
        want = expected.get(name)
        if dest.exists() and want and sha256(dest) != want:
            print(f"{name}: differs from the file the model bundle was built with, downloading again ...")
            dest.unlink()
        if not dest.exists():
            print(f"Downloading {name} ...")
            download(url, dest)
        digest = sha256(dest)
        if want and digest != want:
            mismatched.append(name)
        manifest[name] = {"url": url, "bytes": dest.stat().st_size, "sha256": digest, "license": "Apache-2.0"}
        print(f"  {name}: {dest.stat().st_size/1e6:.1f} MB" + (" (verified against the model bundle)" if want and digest == want else ""))

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

    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {mpath}")
    if mismatched:
        print(f"ERROR: {', '.join(mismatched)} do not match the SHA-256 recorded in the model bundle; "
              "the API will refuse to load it. Check the URLs above or re-export with scripts/train_final.py.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
