"""Cross-validated model comparison under one protocol.

Protocol (identical for every model)
------------------------------------
* Repeated stratified **group** k-fold over annotated person instances (groups =
  near-duplicates + reviewed capture sessions), ``n_repeats x n_splits`` folds.
* Hyper-parameters: inner stratified group k-fold on the training fold only.
* Two test views per fold:
  - ``crop``  — the annotated (ground-truth) person box: isolates classification.
  - ``image`` — whole test images through the real localisation chain
    (person detector -> pose-derived box -> full frame): end-to-end performance.
* The fusion pairing (pose logistic regression + MobileNetV3 probe, equal weights) is
  fixed in advance; other pairings are reported as exploratory only.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from salat_vision.config import Config, get_config
from salat_vision.data.dataset import build_index
from salat_vision.data.splits import assign_groups, inner_train_val, make_cv_splits
from salat_vision.data.voc import load_image_rgb
from salat_vision.evaluation.metrics import (
    ABSTAIN, classification_report_dict, cluster_bootstrap_ci, expected_calibration_error, macro_f1, multiclass_brier)
from salat_vision.features.pose import PoseEstimator, PoseResult, crop_with_padding
from salat_vision.features.pose_features import FEATURE_NAMES, compute_pose_features
from salat_vision.inference.detector import PersonDetector
from salat_vision.labels import VISUAL_CLASSES
from salat_vision.models import classical
from salat_vision.models.rules import fuse, rule_proba

C = len(VISUAL_CLASSES)
BACKBONES = ("mobilenet_v3_large", "resnet18")
PREREGISTERED_FUSION = ("pose_logreg", "cnn_probe_mobilenet_v3_large")

MODEL_INFO = {
    "baseline_majority": ("Baseline", "Majority class of the training fold"),
    "pose_rules": ("Pose", "Training-free geometric rules on MediaPipe pose (a-priori thresholds)"),
    "pose_logreg": ("Pose", "Pose features -> logistic regression"),
    "pose_svm_rbf": ("Pose", "Pose features -> RBF-kernel SVM"),
    "pose_random_forest": ("Pose", "Pose features -> random forest"),
    "pose_mlp": ("Pose", "Pose features -> MLP (1 hidden layer)"),
    "cnn_probe_mobilenet_v3_large": ("CNN", "Frozen MobileNetV3-Large embedding -> logistic regression"),
    "cnn_probe_resnet18": ("CNN", "Frozen ResNet18 embedding -> logistic regression"),
    "cnn_ft_mobilenet_v3_large": ("CNN", "MobileNetV3-Large fine-tuned end-to-end"),
    "cnn_ft_resnet18": ("CNN", "ResNet18 fine-tuned end-to-end"),
    "fusion_pose_cnn": ("Fusion", "Pre-registered: mean of pose logistic regression and MobileNetV3 probe"),
}


# ------------------------------------------------------------------------------ feature cache
@dataclass
class FeatureCache:
    instances: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    crops: list[np.ndarray]
    poses: list[PoseResult]
    X_pose: np.ndarray
    emb: dict[str, dict[str, np.ndarray]]
    images: pd.DataFrame
    det_crops: list[np.ndarray]
    det_poses: list[PoseResult]
    det_X_pose: np.ndarray
    det_emb: dict[str, np.ndarray]
    meta: dict = field(default_factory=dict)


def _iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def localize(rgb: np.ndarray, detector: PersonDetector, pose_est: PoseEstimator, pad: float):
    """Localisation chain used at inference time: detector -> pose box -> full frame."""
    h, w = rgb.shape[:2]
    det = PersonDetector.primary(detector.detect(rgb), (w, h))
    if det is not None:
        box, source = det.as_tuple(), "detector"
    else:
        full_pose = pose_est.detect(rgb)
        b = full_pose.bbox(pad=0.15) if full_pose.detected else None
        if b is not None and (b[2] - b[0]) > 8 and (b[3] - b[1]) > 8:
            box, source = b, "pose"
        else:
            box, source = (0.0, 0.0, float(w), float(h)), "full_image"
    crop, crop_box = crop_with_padding(rgb, box, pad)
    return box, source, crop, crop_box


def build_feature_cache(cfg: Config | None = None, backbones=BACKBONES, log=print) -> FeatureCache:
    import torch

    from salat_vision.models.cnn import build_backbone, extract_embeddings

    cfg = cfg or get_config()
    t0 = time.perf_counter()
    idx = build_index(cfg=cfg)
    groups, _ = assign_groups(idx.images, cfg)
    li = idx.labeled_instances.copy()
    li["group_id"] = li["image_id"].map(groups)
    li["y"] = li["label"].map({c: i for i, c in enumerate(VISUAL_CLASSES)}).astype(int)
    pad = float(cfg.get("dataset", "crop_padding", 0.15))
    pre = cfg.path("pretrained_dir")
    pose_cfg = cfg["pose"]
    min_vis = float(pose_cfg["min_landmark_visibility"])

    images_cache: dict[str, np.ndarray] = {}

    def img(file: str) -> np.ndarray:
        if file not in images_cache:
            images_cache[file] = load_image_rgb(idx.dataset_dir / file)
        return images_cache[file]

    pose_est = PoseEstimator(pre / cfg["models_remote"]["pose_landmarker"]["file"],
                             min_detection_confidence=pose_cfg["min_pose_detection_confidence"],
                             min_presence_confidence=pose_cfg["min_pose_presence_confidence"])
    detector = PersonDetector(pre / cfg["models_remote"]["person_detector"]["file"],
                              score_threshold=cfg["detector"]["score_threshold"], max_results=cfg["detector"]["max_results"])
    crops, poses, feats = [], [], []
    for r in li.itertuples():
        crop, _ = crop_with_padding(img(r.file), (r.xmin, r.ymin, r.xmax, r.ymax), pad)
        pose = pose_est.detect(crop)
        crops.append(crop)
        poses.append(pose)
        feats.append(compute_pose_features(pose, (r.box_w, r.box_h), min_visibility=min_vis))

    # image-level (end-to-end localisation)
    imgs = (li.groupby("image_id").agg(file=("file", "first"), label=("label", "first"), y=("y", "first"),
                                       group_id=("group_id", "first"), n_instances=("instance_id", "size"))
            .reset_index())
    det_crops, det_poses, det_feats, det_rows = [], [], [], []
    for r in imgs.itertuples():
        rgb = img(r.file)
        box, source, crop, _ = localize(rgb, detector, pose_est, pad)
        pose = pose_est.detect(crop)
        gt = li[li["image_id"] == r.image_id]
        best_iou = max(_iou(box, (g.xmin, g.ymin, g.xmax, g.ymax)) for g in gt.itertuples())
        det_crops.append(crop)
        det_poses.append(pose)
        det_feats.append(compute_pose_features(pose, (box[2] - box[0], box[3] - box[1]), min_visibility=min_vis))
        det_rows.append({"loc_source": source, "loc_box": [round(v, 1) for v in box], "loc_best_iou": best_iou,
                         "pose_detected": pose.detected})
    imgs = pd.concat([imgs, pd.DataFrame(det_rows)], axis=1)
    pose_est.close()
    detector.close()

    torch.set_num_threads(4)
    emb, det_emb = {}, {}
    for bb in backbones:
        body, _ = build_backbone(bb, cfg=cfg)
        emb[bb] = {"orig": extract_embeddings(body, crops), "flip": extract_embeddings(body, crops, flip=True)}
        det_emb[bb] = extract_embeddings(body, det_crops)
    meta = {"seconds": time.perf_counter() - t0, "pose_model": cfg["models_remote"]["pose_landmarker"]["file"],
            "detector_model": cfg["models_remote"]["person_detector"]["file"], "crop_padding": pad,
            "n_instances": len(li), "n_images": len(imgs)}
    log(f"feature cache built in {meta['seconds']:.1f}s — {len(li)} instances, {len(imgs)} images")
    return FeatureCache(li.reset_index(drop=True), li["y"].values, li["group_id"].values, crops, poses,
                        np.vstack(feats), emb, imgs, det_crops, det_poses, np.vstack(det_feats), det_emb, meta)


# ------------------------------------------------------------------------------ CV loop
def run_cv(cache: FeatureCache, cfg: Config | None = None, *, models: list[str] | None = None, log=print,
           checkpoint_dir: Path | None = None) -> dict:
    from salat_vision.models.cnn import finetune, predict_proba_torch

    cfg = cfg or get_config()
    models = models or list(MODEL_INFO)
    sp = cfg["splits"]
    seed = cfg.seed
    y, groups = cache.y, cache.groups
    N, NI = len(y), len(cache.images)
    img_groups = cache.images["group_id"].values
    splits = make_cv_splits(y, groups, n_splits=sp["n_splits"], n_repeats=sp["n_repeats"], seed=seed)
    R = sp["n_repeats"]
    oof = {m: np.full((R, N, C), np.nan) for m in models}
    oof_img = {m: np.full((R, NI, C), np.nan) for m in models}
    params: dict[str, list] = {m: [] for m in models}
    histories: dict[str, list] = {}
    timing: dict[str, float] = {m: 0.0 for m in models}
    pose_detected = cache.X_pose[:, FEATURE_NAMES.index("pose_detected")] > 0.5
    det_pose_detected = cache.det_X_pose[:, FEATURE_NAMES.index("pose_detected")] > 0.5

    for s in splits:
        r, f = s["repeat"], s["fold"]
        tr, te = np.array(s["train"]), np.array(s["test"])
        te_img = np.flatnonzero(np.isin(img_groups, np.unique(groups[te])))
        fold_seed = seed + 1000 * r + f
        log(f"[repeat {r} fold {f}] train={len(tr)} test={len(te)} test_images={len(te_img)}")

        def put(name, p_crop, p_img, t):
            oof[name][r, te] = p_crop
            oof_img[name][r, te_img] = p_img
            timing[name] += t

        if "baseline_majority" in models:
            t = time.perf_counter()
            maj = np.bincount(y[tr], minlength=C).argmax()
            p = np.zeros(C)
            p[maj] = 1.0
            put("baseline_majority", np.tile(p, (len(te), 1)), np.tile(p, (len(te_img), 1)), time.perf_counter() - t)

        if "pose_rules" in models:
            t = time.perf_counter()
            put("pose_rules", rule_proba(cache.X_pose[te]), rule_proba(cache.det_X_pose[te_img]), time.perf_counter() - t)

        for name, (pipe, grid) in classical.pose_model_specs(fold_seed).items():
            if name not in models:
                continue
            t = time.perf_counter()
            model, best = classical.fit_with_inner_cv(pipe, grid, cache.X_pose[tr], y[tr], groups[tr], seed=fold_seed)
            put(name, classical.predict_proba_full(model, cache.X_pose[te], C),
                classical.predict_proba_full(model, cache.det_X_pose[te_img], C), time.perf_counter() - t)
            params[name].append(best)

        for bb in BACKBONES:
            name = f"cnn_probe_{bb}"
            if name not in models:
                continue
            t = time.perf_counter()
            pipe, grid = classical.probe_spec(fold_seed)
            Xtr = np.vstack([cache.emb[bb]["orig"][tr], cache.emb[bb]["flip"][tr]])
            ytr = np.concatenate([y[tr], y[tr]])
            gtr = np.concatenate([groups[tr], groups[tr]])
            model, best = classical.fit_with_inner_cv(pipe, grid, Xtr, ytr, gtr, seed=fold_seed)
            put(name, classical.predict_proba_full(model, cache.emb[bb]["orig"][te], C),
                classical.predict_proba_full(model, cache.det_emb[bb][te_img], C), time.perf_counter() - t)
            params[name].append(best)

        for bb in BACKBONES:
            name = f"cnn_ft_{bb}"
            if name not in models:
                continue
            itr, iva = inner_train_val(tr, y, groups, val_fraction=float(sp["inner_val_fraction"]), seed=fold_seed)
            ckpt = (checkpoint_dir / f"{name}_r{r}_f{f}.pt") if checkpoint_dir else None
            res = finetune(bb, [cache.crops[i] for i in itr], y[itr], [cache.crops[i] for i in iva], y[iva],
                           n_classes=C, cfg=cfg, seed=fold_seed, checkpoint_path=ckpt)
            t = time.perf_counter()
            p_crop = predict_proba_torch(res.model, [cache.crops[i] for i in te])
            p_img = predict_proba_torch(res.model, [cache.det_crops[i] for i in te_img])
            put(name, p_crop, p_img, res.train_seconds + time.perf_counter() - t)
            histories.setdefault(name, []).append({"repeat": r, "fold": f, "best_epoch": res.best_epoch,
                                                   "n_train": len(itr), "n_val": len(iva), "history": res.history})
            log(f"    {name}: best epoch {res.best_epoch}, {res.train_seconds:.0f}s")

    if "fusion_pose_cnn" in models and all(m in oof for m in PREREGISTERED_FUSION):
        a, b = PREREGISTERED_FUSION
        for r in range(R):
            pa = oof[a][r].copy()
            pa[~pose_detected] = np.nan  # no pose -> image model only
            oof["fusion_pose_cnn"][r] = fuse(pa, oof[b][r])
            pai = oof_img[a][r].copy()
            pai[~det_pose_detected] = np.nan
            oof_img["fusion_pose_cnn"][r] = fuse(pai, oof_img[b][r])

    return {"splits": splits, "oof": oof, "oof_img": oof_img, "params": params, "histories": histories,
            "timing": timing, "models": models}


# ------------------------------------------------------------------------------ summaries
def _pred(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), ABSTAIN)
    ok = ~np.isnan(p).any(axis=1)
    out[ok] = p[ok].argmax(axis=1)
    return out


def summarize(cache: FeatureCache, cv: dict, *, n_boot: int = 2000, seed: int = 0) -> dict:
    y, groups = cache.y, cache.groups
    yi, gi = cache.images["y"].values, cache.images["group_id"].values
    out = {}
    for m in cv["models"]:
        entry = {"family": MODEL_INFO[m][0], "description": MODEL_INFO[m][1]}
        for view, P, yt, g in (("crop", cv["oof"][m], y, groups), ("image", cv["oof_img"][m], yi, gi)):
            R = P.shape[0]
            if np.isnan(P).all():
                continue
            per_rep = [macro_f1(yt, _pred(P[r]), C) for r in range(R)]
            per_rep_acc = [float((_pred(P[r]) == yt).mean()) for r in range(R)]
            yy = np.tile(yt, R)
            pp = np.concatenate([_pred(P[r]) for r in range(R)])
            gg = np.tile(g, R)
            rep = classification_report_dict(yy, pp, VISUAL_CLASSES)
            lo, hi = cluster_bootstrap_ci(yy, pp, gg, lambda a, b: macro_f1(a, b, C), n_boot=n_boot, seed=seed)
            alo, ahi = cluster_bootstrap_ci(yy, pp, gg, lambda a, b: float((a == b).mean()), n_boot=n_boot, seed=seed)
            prob = np.concatenate([P[r] for r in range(R)])
            entry[view] = {
                "macro_f1_mean": float(np.mean(per_rep)), "macro_f1_std": float(np.std(per_rep)),
                "macro_f1_per_repeat": per_rep, "macro_f1_ci95": [lo, hi],
                "accuracy_mean": float(np.mean(per_rep_acc)), "accuracy_ci95": [alo, ahi],
                "pooled": rep,
                "calibration": expected_calibration_error(yy, prob, n_bins=5) if m != "baseline_majority" else None,
                "brier": multiclass_brier(yy, prob) if m != "baseline_majority" else None,
            }
        entry["train_seconds_total"] = float(cv["timing"].get(m, 0.0))
        entry["best_params"] = cv["params"].get(m, [])
        out[m] = entry
    return out


def oof_table(cache: FeatureCache, cv: dict) -> pd.DataFrame:
    rows = []
    for view, key, frame, idcol in (("crop", "oof", cache.instances, "instance_id"), ("image", "oof_img", cache.images, "image_id")):
        for m in cv["models"]:
            P = cv[key][m]
            for r in range(P.shape[0]):
                pred = _pred(P[r])
                for i, rid in enumerate(frame[idcol].values):
                    row = {"view": view, "model": m, "repeat": r, "id": rid, "group_id": frame["group_id"].values[i],
                           "true": VISUAL_CLASSES[int(frame["y"].values[i])],
                           "pred": VISUAL_CLASSES[pred[i]] if pred[i] >= 0 else "abstain"}
                    for c, v in zip(VISUAL_CLASSES, P[r, i]):
                        row[f"p_{c}"] = None if np.isnan(v) else round(float(v), 5)
                    rows.append(row)
    return pd.DataFrame(rows)


def save_cv(cache: FeatureCache, cv: dict, summary: dict, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"summary": out_dir / "cv_results.json", "oof": out_dir / "oof_predictions.csv",
             "histories": out_dir / "training_histories.json", "localization": out_dir / "localization_eval.csv"}
    paths["summary"].write_text(json.dumps({"protocol": __doc__.strip(), "classes": VISUAL_CLASSES,
                                            "preregistered_fusion": list(PREREGISTERED_FUSION),
                                            "feature_cache": cache.meta, "models": summary}, indent=2), encoding="utf-8")
    oof_table(cache, cv).to_csv(paths["oof"], index=False)
    paths["histories"].write_text(json.dumps(cv["histories"], indent=1), encoding="utf-8")
    cache.images.drop(columns=["file"]).to_csv(paths["localization"], index=False)
    return paths
