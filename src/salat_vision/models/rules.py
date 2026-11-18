"""Training-free, interpretable posture rules on pose geometry.

The thresholds come from body mechanics, fixed *before* any evaluation and never tuned
on the dataset:

* legs straight (knee angle > 140 deg)  -> standing or bowing
  - torso within 45 deg of vertical      -> standing (qiyam)
  - otherwise                            -> bowing (ruku)
* legs bent                              -> sitting or prostration
  - torso tilted more than 55 deg        -> prostration (sujud)
  - otherwise                            -> sitting (julus)

Hard thresholds are softened with logistic ramps (10 deg scale) so the output is a
probability-like score that can be fused or decoded over time. No pose -> abstain (NaN).
"""

from __future__ import annotations

import math

import numpy as np

from salat_vision.features.pose_features import FEATURE_NAMES
from salat_vision.labels import VISUAL_CLASSES

KNEE_STRAIGHT_DEG = 140.0
TORSO_BOW_DEG = 45.0
TORSO_FOLD_DEG = 55.0
RAMP_DEG = 10.0

_F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _first(row: np.ndarray, *names: str) -> float:
    for n in names:
        v = float(row[_F[n]])
        if not math.isnan(v):
            return v
    return float("nan")


def rule_proba(X: np.ndarray) -> np.ndarray:
    """Rows of pose features -> (N, 4) scores in VISUAL_CLASSES order; NaN rows = abstain."""
    out = np.full((len(X), len(VISUAL_CLASSES)), np.nan)
    for i, row in enumerate(X):
        if row[_F["pose_detected"]] < 0.5:
            continue
        knee = _first(row, "knee_angle_3d", "knee_angle_2d")
        torso = _first(row, "torso_incl_3d", "torso_incl_2d")
        p_straight = 0.5 if math.isnan(knee) else _sig((knee - KNEE_STRAIGHT_DEG) / RAMP_DEG)
        p_upright = 0.5 if math.isnan(torso) else _sig((TORSO_BOW_DEG - torso) / RAMP_DEG)
        p_folded = 0.5 if math.isnan(torso) else _sig((torso - TORSO_FOLD_DEG) / RAMP_DEG)
        scores = {
            "qiyam": p_straight * p_upright,
            "ruku": p_straight * (1 - p_upright),
            "sujud": (1 - p_straight) * p_folded,
            "julus": (1 - p_straight) * (1 - p_folded),
        }
        out[i] = [scores[c] for c in VISUAL_CLASSES]
    return out


def fuse(p_pose: np.ndarray, p_img: np.ndarray, *, w_pose: float = 0.5) -> np.ndarray:
    """Late fusion: weighted mean where the pose model produced scores, image model elsewhere."""
    out = np.array(p_img, dtype=np.float64, copy=True)
    ok = ~np.isnan(p_pose).any(axis=1)
    out[ok] = w_pose * p_pose[ok] + (1.0 - w_pose) * p_img[ok]
    return out
