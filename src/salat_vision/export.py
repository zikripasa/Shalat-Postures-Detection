"""Train the selected pipeline on all labelled data and export a self-contained bundle."""

from __future__ import annotations

import datetime as dt
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from salat_vision import __version__
from salat_vision.config import Config, get_config
from salat_vision.experiments import FeatureCache, build_feature_cache
from salat_vision.features.pose_features import FEATURE_NAMES
from salat_vision.inference.bundle import sha256_file
from salat_vision.labels import PHASE_LABELS_ID, PHASES, VISUAL_CLASSES, VISUAL_LABELS_ID
from salat_vision.models import classical


def export_onnx_embedder(backbone_name: str, path: Path, cfg: Config, size: int = 224) -> dict:
    import torch

    from salat_vision.models.cnn import build_backbone

    body, dim = build_backbone(backbone_name, cfg=cfg)
    body.eval()
    x = torch.zeros(1, 3, size, size)
    kwargs = dict(input_names=["input"], output_names=["embedding"],
                  dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(body, (x,), str(path), dynamo=False, **kwargs)
        exporter = "torchscript"
    except Exception:  # pragma: no cover - fall back to the dynamo exporter
        torch.onnx.export(body, (x,), str(path), dynamo=True, **kwargs)
        exporter = "dynamo"
    return {"dim": dim, "exporter": exporter, "opset": 17}


def onnx_parity(backbone_name: str, onnx_path: Path, crops: list[np.ndarray], cfg: Config, size: int = 224) -> dict:
    import onnxruntime as ort

    from salat_vision.models.cnn import build_backbone, extract_embeddings, preprocess_batch

    body, _ = build_backbone(backbone_name, cfg=cfg)
    ref = extract_embeddings(body, crops, size=size)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = np.vstack([sess.run(None, {"input": preprocess_batch([c], size)})[0] for c in crops])
    cos = (ref * got).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1))
    return {"max_abs_diff": float(np.abs(ref - got).max()), "min_cosine": float(cos.min()), "n": len(crops)}


def train_and_export(cfg: Config | None = None, cache: FeatureCache | None = None, log=print) -> Path:
    import joblib
    import sklearn

    cfg = cfg or get_config()
    fm = cfg["final_model"]
    out = cfg.path("models_dir") / fm["bundle_name"]
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cache = cache or build_feature_cache(cfg, backbones=(fm["image_backbone"],) if fm.get("image_backbone") else ())
    y, groups, seed = cache.y, cache.groups, cfg.seed
    files, training = [], {}

    pose_name = fm.get("pose_classifier")
    if pose_name:
        pipe, grid = classical.pose_model_specs(seed)[pose_name]
        model, best = classical.fit_with_inner_cv(pipe, grid, cache.X_pose, y, groups, seed=seed)
        joblib.dump(model, tmp / "pose_classifier.joblib")
        training["pose_classifier"] = {"type": pose_name, "best_params": best, "n_train": int(len(y))}
        log(f"pose classifier {pose_name}: {best}")

    image_model = None
    bb = fm.get("image_backbone")
    if bb:
        pipe, grid = classical.probe_spec(seed)
        X = np.vstack([cache.emb[bb]["orig"], cache.emb[bb]["flip"]])
        model, best = classical.fit_with_inner_cv(pipe, grid, X, np.concatenate([y, y]), np.concatenate([groups, groups]), seed=seed)
        joblib.dump(model, tmp / "image_classifier.joblib")
        onnx_name = f"cnn_embedder_{bb}.onnx"
        info = export_onnx_embedder(bb, tmp / onnx_name, cfg, size=int(cfg["cnn"]["image_size"]))
        parity = onnx_parity(bb, tmp / onnx_name, cache.crops, cfg, size=int(cfg["cnn"]["image_size"]))
        log(f"ONNX parity: {parity}")
        if parity["min_cosine"] < 0.9999:
            raise RuntimeError(f"ONNX export differs from PyTorch: {parity}")
        training["image_classifier"] = {"type": f"logistic regression on frozen {bb} embedding", "best_params": best,
                                        "n_train": int(len(y)), "train_augmentation": "horizontal flip (x2)",
                                        "onnx": info, "onnx_parity": parity}
        image_model = {"embedder": onnx_name, "classifier": "image_classifier.joblib", "backbone": bb,
                       "embedding_dim": info["dim"], "input_size": int(cfg["cnn"]["image_size"]),
                       "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
                       "resize": "aspect-preserving letterbox, fill (124,116,104)", "threads": 2}

    for name in sorted(p.name for p in tmp.iterdir()):
        files.append({"name": name, "bytes": (tmp / name).stat().st_size, "sha256": sha256_file(tmp / name)})

    pre = cfg.path("pretrained_dir")
    mr = cfg["models_remote"]
    third_party = [
        {"role": "pose_landmarker", "name": mr["pose_landmarker"]["file"], "url": mr["pose_landmarker"]["url"],
         "sha256": sha256_file(pre / mr["pose_landmarker"]["file"]), "license": "Apache-2.0 (Google MediaPipe)"},
        {"role": "person_detector", "name": mr["person_detector"]["file"], "url": mr["person_detector"]["url"],
         "sha256": sha256_file(pre / mr["person_detector"]["file"]), "license": "Apache-2.0 (Google MediaPipe)"},
    ]
    cv_path = cfg.path("artifacts_dir") / "metrics" / "cv_results.json"
    cv_metrics = None
    if cv_path.is_file():
        cvr = json.loads(cv_path.read_text())
        sel = fm.get("cv_model_key")
        if sel and sel in cvr["models"]:
            e = cvr["models"][sel]
            cv_metrics = {
                "model_key": sel,
                "protocol": f"{cfg['splits']['n_repeats']}x repeated stratified group {cfg['splits']['n_splits']}-fold CV",
                "crop": {k: e["crop"][k] for k in ("macro_f1_mean", "macro_f1_std", "macro_f1_ci95", "accuracy_mean", "accuracy_ci95")},
                "image": {k: e["image"][k] for k in ("macro_f1_mean", "macro_f1_std", "macro_f1_ci95", "accuracy_mean", "accuracy_ci95")},
                "per_class_crop": e["crop"]["pooled"]["per_class"],
                "confusion_matrix_crop": e["crop"]["pooled"]["confusion_matrix"],
                "confusion_matrix_image": e["image"]["pooled"]["confusion_matrix"],
                "ece_crop": (e["crop"].get("calibration") or {}).get("ece"),
            }
    import mediapipe
    import onnxruntime
    import torch

    manifest = {
        "name": "Salat Vision posture model",
        "version": fm["version"],
        "package_version": __version__,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "classes": VISUAL_CLASSES,
        "class_labels_id": VISUAL_LABELS_ID,
        "phases": PHASES,
        "phase_labels_id": PHASE_LABELS_ID,
        "pipeline": {
            "crop_padding": float(cfg["dataset"]["crop_padding"]),
            "localization": ["person_detector (EfficientDet-Lite0, COCO person)", "pose-landmark box", "full frame"],
            "pose_estimator": {"min_detection_confidence": cfg["pose"]["min_pose_detection_confidence"],
                               "min_presence_confidence": cfg["pose"]["min_pose_presence_confidence"]},
            "person_detector": {"score_threshold": cfg["detector"]["score_threshold"], "max_results": cfg["detector"]["max_results"]},
            "pose_features": FEATURE_NAMES,
            "min_landmark_visibility": float(cfg["pose"]["min_landmark_visibility"]),
            "pose_classifier": "pose_classifier.joblib" if pose_name else None,
            "image_model": image_model,
            "fusion": {"rule": "weighted mean of class probabilities; image model alone when no pose is found",
                       "w_pose": float(fm.get("fusion_w_pose", 0.5))},
            "uncertainty_threshold": float(cfg["inference"]["uncertainty_threshold"]),
        },
        "training": {**training, "dataset": {"instances": int(len(y)), "images": int(len(cache.images)),
                                              "groups": int(len(np.unique(groups))),
                                              "per_class": {c: int((y == i).sum()) for i, c in enumerate(VISUAL_CLASSES)},
                                              "source": "Salat-Postures-Dataset (GitHub sample folder)"}},
        "cv_metrics": cv_metrics,
        "files": files,
        "third_party": third_party,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
                        "torch": torch.__version__, "onnxruntime": onnxruntime.__version__, "mediapipe": mediapipe.__version__,
                        "scikit_learn": sklearn.__version__, "numpy": np.__version__},
        "intended_use": "Research/education demo: recognising visible prayer postures in images and short videos.",
        "out_of_scope": ["Judging whether a prayer is religiously valid", "Identity recognition", "Surveillance"],
        "limitations": [
            "Trained on 40 annotated people (29 independent groups); estimates are high-variance.",
            "Iktidal, sitting between prostrations and tasyahud are inferred from posture order, not seen directly.",
            "Pose detection fails more often in prostration (heads down, folded bodies).",
            "Not validated on real continuous prayer videos.",
        ],
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    log(f"bundle written to {out}")
    return out
