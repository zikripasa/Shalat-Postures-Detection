import json

import numpy as np
import pytest

from salat_vision.data.voc import load_image_rgb
from salat_vision.inference.bundle import BundleError, ModelBundle
from salat_vision.labels import VISUAL_CLASSES

from .conftest import make_video, requires_data, requires_model

pytestmark = [pytest.mark.slow, requires_model]


def test_bundle_manifest_and_checksums(cfg):
    b = ModelBundle.load(cfg=cfg, verify=True)
    assert b.classes == VISUAL_CLASSES
    assert b.manifest["pipeline"]["fusion"]["w_pose"] == pytest.approx(0.5)
    assert b.manifest["training"]["image_classifier"]["onnx_parity"]["min_cosine"] > 0.9999
    assert b.manifest["cv_metrics"]["crop"]["macro_f1_mean"] > 0


def test_bundle_detects_tampering(cfg, tmp_path):
    import shutil

    src = cfg.path("models_dir") / cfg["final_model"]["bundle_name"]
    dst = tmp_path / "bundle"
    shutil.copytree(src, dst)
    with open(dst / "pose_classifier.joblib", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(BundleError):
        ModelBundle.load(dst, cfg=cfg, verify=True)


@requires_data
def test_predict_real_image(predictor, sample_image_path):
    pred = predictor.predict(load_image_rgb(sample_image_path))
    assert pred.probs is not None and pred.probs.shape == (4,)
    assert pred.probs.sum() == pytest.approx(1.0, abs=1e-6)
    d = pred.to_dict()
    json.dumps(d)  # plain JSON types only
    assert d["localization"] in {"detector", "pose", "full_image"}
    assert d["landmarks"] is None or len(d["landmarks"]) == 33


def test_predict_blank_frame_is_not_confident(predictor):
    pred = predictor.predict(np.zeros((360, 640, 3), np.uint8))
    assert pred.loc_source == "full_image"
    assert not pred.pose.detected


@requires_data
def test_video_pipeline_end_to_end(predictor, cfg, tmp_path, sample_image_path):
    from salat_vision.inference.video import analyze_video, probe_video

    img = load_image_rgb(sample_image_path)
    h, w = img.shape[:2]
    s = 480 / max(h, w)
    import cv2

    frame = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2))
    video = make_video(tmp_path / "clip.mp4", [frame] * 20 + [np.zeros_like(frame)] * 10, fps=10)
    info = probe_video(video)
    assert info["duration"] == pytest.approx(3.0, abs=0.15)
    out = tmp_path / "annotated.mp4"
    res = analyze_video(video, predictor, cfg, sample_fps=5, annotate_path=out)
    json.dumps(res)
    segs = res["segments"]
    assert segs[0]["start"] == 0.0
    assert segs[-1]["end"] == pytest.approx(res["video"]["duration_seconds"], abs=0.25)
    assert res["performance"]["frames_analyzed"] == pytest.approx(15, abs=1)
    assert out.is_file() and probe_video(out)["width"] > 0
