import math

import numpy as np
import pytest

from salat_vision.evaluation.metrics import ABSTAIN, classification_report_dict, cluster_bootstrap_ci, expected_calibration_error, macro_f1
from salat_vision.features.pose import LM, PoseResult
from salat_vision.features.pose_features import FEATURE_NAMES, compute_pose_features
from salat_vision.labels import VISUAL_CLASSES
from salat_vision.models.cnn import letterbox, preprocess_batch
from salat_vision.models.rules import fuse, rule_proba

F = {n: i for i, n in enumerate(FEATURE_NAMES)}

# side-view stick figures in metres (y axis points down, as in MediaPipe world landmarks)
POSES = {
    "qiyam": {"nose": (0.05, -0.7), "shoulder": (0, -0.5), "hip": (0, 0), "knee": (0, 0.45), "ankle": (0, 0.9), "wrist": (0.1, -0.15)},
    "ruku": {"nose": (0.65, 0.05), "shoulder": (0.5, 0.0), "hip": (0, 0), "knee": (0.02, 0.45), "ankle": (0, 0.9), "wrist": (0.25, 0.4)},
    "sujud": {"nose": (0.55, 0.45), "shoulder": (0.42, 0.3), "hip": (0, 0), "knee": (0.08, 0.45), "ankle": (-0.35, 0.47), "wrist": (0.6, 0.45)},
    "julus": {"nose": (0.05, -0.68), "shoulder": (0, -0.5), "hip": (0, 0), "knee": (0.45, 0.1), "ankle": (0.0, 0.16), "wrist": (0.3, 0.0)},
}


def make_pose(kind: str, visibility: float = 0.95) -> PoseResult:
    p = POSES[kind]
    world = np.zeros((33, 3), np.float32)
    for side, dz in (("left", -0.1), ("right", 0.1)):
        for part in ("shoulder", "hip", "knee", "ankle", "wrist"):
            x, y = p[part]
            world[LM[f"{side}_{part}"]] = (x, y, dz)
    world[LM["nose"]] = (*p["nose"], 0)
    lms = np.zeros((33, 5), np.float32)
    lms[:, 0] = 0.5 + world[:, 0] * 0.4
    lms[:, 1] = 0.5 + world[:, 1] * 0.4
    lms[:, 3] = visibility
    lms[:, 4] = visibility
    return PoseResult(True, (400, 400), lms, world)


def test_standing_geometry():
    f = compute_pose_features(make_pose("qiyam"), (100, 300))
    assert f[F["pose_detected"]] == 1
    assert f[F["torso_incl_3d"]] < 10
    assert f[F["knee_angle_3d"]] > 170
    assert f[F["head_rel_y"]] < 0  # head above hips
    assert math.isclose(f[F["box_log_aspect"]], math.log(3), rel_tol=1e-6)


@pytest.mark.parametrize("kind", VISUAL_CLASSES)
def test_rules_classify_canonical_postures(kind):
    p = rule_proba(compute_pose_features(make_pose(kind), (200, 200))[None])[0]
    assert VISUAL_CLASSES[int(np.argmax(p))] == kind
    assert math.isclose(p.sum(), 1.0, rel_tol=1e-6)


def test_missing_pose_yields_nan_and_abstention():
    f = compute_pose_features(None, (50, 100))
    assert f[F["pose_detected"]] == 0 and not math.isnan(f[F["box_log_aspect"]])
    assert np.isnan(f[F["torso_incl_3d"]])
    assert np.isnan(rule_proba(f[None])).all()


def test_low_visibility_landmarks_are_missing():
    f = compute_pose_features(make_pose("qiyam", visibility=0.1), (100, 300), min_visibility=0.35)
    assert np.isnan(f[F["knee_angle_3d"]]) and np.isnan(f[F["torso_incl_3d"]])


def test_fusion_falls_back_to_image_model():
    p_img = np.array([[0.1, 0.2, 0.3, 0.4], [0.7, 0.1, 0.1, 0.1]])
    p_pose = np.array([[np.nan] * 4, [0.1, 0.1, 0.1, 0.7]])
    out = fuse(p_pose, p_img, w_pose=0.5)
    np.testing.assert_allclose(out[0], p_img[0])
    np.testing.assert_allclose(out[1], [0.4, 0.1, 0.1, 0.4])


def test_letterbox_preserves_aspect_and_size():
    img = np.zeros((300, 100, 3), np.uint8)
    img[:] = 255
    sq = letterbox(img, 224)
    assert sq.shape == (224, 224, 3)
    cols = np.flatnonzero((sq == 255).all(axis=2).any(axis=0))
    assert 70 <= len(cols) <= 80  # 224 * 100/300 ~= 75 px wide content
    x = preprocess_batch([img], 224)
    assert x.shape == (1, 3, 224, 224) and x.dtype == np.float32


def test_metrics_with_abstention_and_ci():
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    pred = np.array([0, 1, 2, 3, 0, 1, ABSTAIN, 0])
    rep = classification_report_dict(y, pred, VISUAL_CLASSES)
    assert rep["accuracy"] == 0.75 and rep["abstain_rate"] == 0.125
    # rows = true class; the last column counts abstentions separately (never as a class)
    assert rep["confusion_matrix"][2] == [0, 0, 1, 0, 1]
    assert rep["confusion_matrix"][3] == [1, 0, 0, 1, 0]
    g = np.arange(8)
    lo, hi = cluster_bootstrap_ci(y, pred, g, lambda a, b: macro_f1(a, b, 4), n_boot=200)
    assert 0 <= lo <= macro_f1(y, pred, 4) <= hi <= 1


def test_ece_perfectly_calibrated_is_zero():
    y = np.array([0, 1])
    proba = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    assert expected_calibration_error(y, proba)["ece"] == pytest.approx(0.0)
