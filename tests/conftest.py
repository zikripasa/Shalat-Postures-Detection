from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GLOG_minloglevel", "2")

from salat_vision.config import get_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return get_config()


def _has_data() -> bool:
    d = get_config().path("dataset_dir")
    return d is not None and d.is_dir() and any(d.rglob("*.xml"))


def _has_model() -> bool:
    cfg = get_config()
    return (cfg.path("models_dir") / cfg["final_model"]["bundle_name"] / "manifest.json").is_file() and \
        (cfg.path("pretrained_dir") / cfg["models_remote"]["pose_landmarker"]["file"]).is_file()


HAS_DATA = _has_data()
HAS_MODEL = _has_model()

requires_data = pytest.mark.skipif(not HAS_DATA, reason="dataset not extracted (scripts/extract_dataset.py)")
requires_model = pytest.mark.skipif(not HAS_MODEL, reason="model bundle / pretrained models missing")


@pytest.fixture(scope="session")
def sample_image_path(cfg):
    if not HAS_DATA:
        pytest.skip("dataset not available")
    return cfg.path("dataset_dir") / "Sample-images-with-annotations" / "ck2upgs2ttpwm07575jjfctev.jpeg"


@pytest.fixture(scope="session")
def predictor():
    if not HAS_MODEL:
        pytest.skip("model bundle not available")
    from salat_vision.inference.predictor import PosturePredictor

    p = PosturePredictor()
    yield p
    p.close()


def make_video(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> Path:
    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    writer = imageio_ffmpeg.write_frames(str(path), (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                                         macro_block_size=2, output_params=["-preset", "ultrafast"])
    writer.send(None)
    for f in frames:
        writer.send(np.ascontiguousarray(f))
    writer.close()
    return path
