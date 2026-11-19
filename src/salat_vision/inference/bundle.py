"""Model bundle: a versioned folder with a manifest, checksums and every artefact
needed for inference (no training code, no PyTorch).

Layout::

    models/salat_vision_v1/
      manifest.json                         classes, preprocessing, metrics, checksums
      cnn_embedder_mobilenet_v3_large.onnx  frozen ImageNet backbone (ONNX Runtime)
      image_classifier.joblib               scaler + logistic regression on embeddings
      pose_classifier.joblib                imputer + scaler + logistic regression on pose features

Third-party MediaPipe models (pose landmarker, person detector) are referenced by file
name and SHA-256 and resolved from ``paths.pretrained_dir``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from salat_vision.config import Config, get_config

DEFAULT_BUNDLE = "salat_vision_v1"


class BundleError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ModelBundle:
    root: Path
    manifest: dict
    pretrained_dir: Path

    @classmethod
    def load(cls, bundle_dir: str | Path | None = None, cfg: Config | None = None, verify: bool = True) -> "ModelBundle":
        cfg = cfg or get_config()
        root = Path(bundle_dir) if bundle_dir else cfg.path("models_dir") / DEFAULT_BUNDLE
        mpath = root / "manifest.json"
        if not mpath.is_file():
            raise BundleError(f"No model bundle at {root} (manifest.json missing). Run scripts/train_final.py.")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        b = cls(root, manifest, cfg.path("pretrained_dir"))
        if verify:
            b.verify()
        return b

    def file(self, name: str) -> Path:
        return self.root / name

    def pretrained(self, name: str) -> Path:
        return self.pretrained_dir / name

    def verify(self) -> None:
        """Check every bundled and third-party file against the manifest checksums."""
        for entry in self.manifest.get("files", []):
            p = self.file(entry["name"])
            if not p.is_file():
                raise BundleError(f"Bundle file missing: {p}")
            if sha256_file(p) != entry["sha256"]:
                raise BundleError(f"Checksum mismatch for {p.name}")
        for entry in self.manifest.get("third_party", []):
            p = self.pretrained(entry["name"])
            if not p.is_file():
                raise BundleError(f"Third-party model missing: {p}. Run scripts/download_models.py.")
            if entry.get("sha256") and sha256_file(p) != entry["sha256"]:
                raise BundleError(f"Checksum mismatch for third-party model {p.name}")

    @property
    def version(self) -> str:
        return self.manifest.get("version", "unknown")

    @property
    def classes(self) -> list[str]:
        return self.manifest["classes"]
