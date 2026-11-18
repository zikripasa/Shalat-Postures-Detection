"""Engineered, view-tolerant pose features.

Every feature is invariant to image translation and scale. Angles use MediaPipe's metric
*world* landmarks (y axis points down, verified on standing people) where possible, so they
are less sensitive to the camera position than 2-D angles. Landmarks whose visibility is
below a threshold are treated as missing (``NaN``); the downstream pipeline imputes them and
adds explicit missing-value indicators.
"""

from __future__ import annotations

import math

import numpy as np

from salat_vision.features.pose import LM, PoseResult

FEATURE_NAMES: list[str] = [
    "pose_detected",
    # 3-D (world) angles, degrees
    "torso_incl_3d",       # hip->shoulder vs. up: 0 upright, 90 horizontal, >90 head below hips
    "hip_angle_3d",        # shoulder-hip-knee: 180 straight body line, small = folded
    "knee_angle_3d",       # hip-knee-ankle: 180 straight leg, small = deeply bent
    "thigh_incl_3d",       # hip->knee vs. down: 0 vertical thigh, 90 horizontal thigh
    "shank_incl_3d",       # knee->ankle vs. down: 0 vertical shin, 90 shin on the floor
    # 2-D (image) angles, degrees
    "torso_incl_2d",
    "knee_angle_2d",
    # relative heights in the image (positive = below the mid-hip), / landmark-box diagonal
    "head_rel_y",
    "shoulder_rel_y",
    "wrist_rel_y",
    "knee_rel_y",
    "ankle_rel_y",
    # horizontal spread (view-symmetric), / landmark-box diagonal
    "head_rel_dx",
    "shoulder_rel_dx",
    # silhouette
    "lm_box_log_aspect",   # log(height / width) of the visible-landmark box
    # quality
    "vis_upper",
    "vis_lower",
    # person box (not pose): available even when no pose is found
    "box_log_aspect",
]

_UPPER = [LM["nose"], LM["left_shoulder"], LM["right_shoulder"], LM["left_elbow"], LM["right_elbow"],
          LM["left_wrist"], LM["right_wrist"]]
_LOWER = [LM["left_hip"], LM["right_hip"], LM["left_knee"], LM["right_knee"], LM["left_ankle"], LM["right_ankle"]]

UP = np.array([0.0, -1.0, 0.0])
DOWN = -UP
UP2 = np.array([0.0, -1.0])
DOWN2 = -UP2


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return float("nan")
    c = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
    return math.degrees(math.acos(c))


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` formed by a-b-c (degrees)."""
    return _angle(a - b, c - b)


def _mid(points: np.ndarray, vis: np.ndarray, i: int, j: int, thr: float) -> np.ndarray | None:
    """Visibility-aware midpoint of a left/right landmark pair (falls back to the visible side)."""
    ok_i, ok_j = vis[i] >= thr, vis[j] >= thr
    if ok_i and ok_j:
        return (points[i] + points[j]) / 2.0
    if ok_i:
        return points[i]
    if ok_j:
        return points[j]
    return None


def _side_mean(values: list[tuple[float, float]]) -> float:
    """Visibility-weighted mean of per-side values [(value, weight), ...] ignoring NaNs."""
    vals = [(v, w) for v, w in values if not math.isnan(v) and w > 0]
    if not vals:
        return float("nan")
    tot = sum(w for _, w in vals)
    return float(sum(v * w for v, w in vals) / tot)


def compute_pose_features(pose: PoseResult | None, box_wh: tuple[float, float] | None, *, min_visibility: float = 0.35) -> np.ndarray:
    """Feature vector (``len(FEATURE_NAMES)``) for one person; ``NaN`` marks missing values."""
    f = {name: float("nan") for name in FEATURE_NAMES}
    if box_wh is not None and box_wh[0] > 0 and box_wh[1] > 0:
        f["box_log_aspect"] = math.log(box_wh[1] / box_wh[0])
    if pose is None or not pose.detected or pose.landmarks is None:
        f["pose_detected"] = 0.0
        return np.array([f[n] for n in FEATURE_NAMES], dtype=np.float32)

    f["pose_detected"] = 1.0
    lm = pose.landmarks
    vis = lm[:, 3]
    thr = min_visibility
    xy = pose.pixel_xy().astype(np.float64)
    f["vis_upper"] = float(vis[_UPPER].mean())
    f["vis_lower"] = float(vis[_LOWER].mean())

    # ---- 2-D geometry -----------------------------------------------------------------
    visible = vis >= thr
    pts = xy[visible] if visible.sum() >= 4 else xy
    span = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.hypot(*span)) or 1.0
    f["lm_box_log_aspect"] = math.log(max(span[1], 1.0) / max(span[0], 1.0))
    S2 = _mid(xy, vis, LM["left_shoulder"], LM["right_shoulder"], thr)
    H2 = _mid(xy, vis, LM["left_hip"], LM["right_hip"], thr)
    K2 = _mid(xy, vis, LM["left_knee"], LM["right_knee"], thr)
    A2 = _mid(xy, vis, LM["left_ankle"], LM["right_ankle"], thr)
    W2 = _mid(xy, vis, LM["left_wrist"], LM["right_wrist"], thr)
    N2 = xy[LM["nose"]] if vis[LM["nose"]] >= thr else None
    if H2 is not None:
        if S2 is not None:
            f["torso_incl_2d"] = _angle(S2 - H2, UP2)
            f["shoulder_rel_y"] = (S2[1] - H2[1]) / diag
            f["shoulder_rel_dx"] = abs(S2[0] - H2[0]) / diag
        if N2 is not None:
            f["head_rel_y"] = (N2[1] - H2[1]) / diag
            f["head_rel_dx"] = abs(N2[0] - H2[0]) / diag
        if W2 is not None:
            f["wrist_rel_y"] = (W2[1] - H2[1]) / diag
        if K2 is not None:
            f["knee_rel_y"] = (K2[1] - H2[1]) / diag
        if A2 is not None:
            f["ankle_rel_y"] = (A2[1] - H2[1]) / diag
    knee2 = []
    for hip, knee, ankle in (("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")):
        w = min(vis[LM[hip]], vis[LM[knee]], vis[LM[ankle]])
        if w >= thr:
            knee2.append((_joint_angle(xy[LM[hip]], xy[LM[knee]], xy[LM[ankle]]), w))
    f["knee_angle_2d"] = _side_mean(knee2)

    # ---- 3-D (world) geometry -----------------------------------------------------------
    W = pose.world
    if W is not None:
        W = W.astype(np.float64)
        S3 = _mid(W, vis, LM["left_shoulder"], LM["right_shoulder"], thr)
        H3 = _mid(W, vis, LM["left_hip"], LM["right_hip"], thr)
        K3 = _mid(W, vis, LM["left_knee"], LM["right_knee"], thr)
        A3 = _mid(W, vis, LM["left_ankle"], LM["right_ankle"], thr)
        if S3 is not None and H3 is not None:
            f["torso_incl_3d"] = _angle(S3 - H3, UP)
        if H3 is not None and K3 is not None:
            f["thigh_incl_3d"] = _angle(K3 - H3, DOWN)
        if K3 is not None and A3 is not None:
            f["shank_incl_3d"] = _angle(A3 - K3, DOWN)
        hips, knees = [], []
        for side in ("left", "right"):
            s, h, k, a = (LM[f"{side}_{p}"] for p in ("shoulder", "hip", "knee", "ankle"))
            wh = min(vis[s], vis[h], vis[k])
            if wh >= thr:
                hips.append((_joint_angle(W[s], W[h], W[k]), wh))
            wk = min(vis[h], vis[k], vis[a])
            if wk >= thr:
                knees.append((_joint_angle(W[h], W[k], W[a]), wk))
        f["hip_angle_3d"] = _side_mean(hips)
        f["knee_angle_3d"] = _side_mean(knees)

    return np.array([f[n] for n in FEATURE_NAMES], dtype=np.float32)
