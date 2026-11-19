#!/usr/bin/env python
"""Export portfolio/assets/data/deck_data.json — the values behind every native chart,
table and timeline in the PPTX deck — from artifacts/metrics (run make_figures.py first)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "artifacts" / "metrics"
NAMES = {"fusion_pose_cnn": "Fusion (selected)", "pose_logreg": "Pose + logistic reg.", "pose_mlp": "Pose + MLP",
         "cnn_ft_mobilenet_v3_large": "MobileNetV3 fine-tuned", "cnn_probe_mobilenet_v3_large": "MobileNetV3 frozen probe",
         "pose_random_forest": "Pose + random forest", "pose_svm_rbf": "Pose + RBF SVM", "cnn_ft_resnet18": "ResNet18 fine-tuned",
         "pose_rules": "Pose rules (no training)", "cnn_probe_resnet18": "ResNet18 frozen probe", "baseline_majority": "Majority baseline"}


def main() -> int:
    cv = json.loads((M / "cv_results.json").read_text())
    cmp = pd.read_csv(M / "model_comparison.csv").sort_values("crop_macro_f1", ascending=False)
    audit = json.loads((M / "dataset_audit.json").read_text())
    rt = json.loads((M / "runtime_benchmark.json").read_text())
    tb = json.loads((M / "temporal_benchmark.json").read_text())
    pv = json.loads((M / "pose_variant_benchmark.json").read_text())
    demo = json.loads((ROOT / "portfolio" / "assets" / "data" / "posture_timeline_segments.json").read_text())
    loc = pd.read_csv(M / "localization_eval.csv")
    sel = "fusion_pose_cnn"
    e = cv["models"][sel]
    deck = {
        "selected": sel,
        "models": [{"key": r.model, "name": NAMES[r.model], "crop": round(r.crop_macro_f1, 3), "crop_lo": round(r.crop_macro_f1_ci_lo, 3),
                    "crop_hi": round(r.crop_macro_f1_ci_hi, 3), "image": round(r.image_macro_f1, 3), "sd": round(r.crop_macro_f1_sd, 3)}
                   for r in cmp.itertuples()],
        "selected_metrics": {"crop_f1": e["crop"]["macro_f1_mean"], "crop_ci": e["crop"]["macro_f1_ci95"], "image_f1": e["image"]["macro_f1_mean"],
                             "image_ci": e["image"]["macro_f1_ci95"], "crop_acc": e["crop"]["accuracy_mean"], "ece": e["crop"]["calibration"]["ece"]},
        "per_class": e["crop"]["pooled"]["per_class"],
        "confusion_crop": [r[:4] for r in e["crop"]["pooled"]["confusion_matrix"]],
        "confusion_image": [r[:4] for r in e["image"]["pooled"]["confusion_matrix"]],
        "dataset": {"images": audit["n_image_files"], "instances": audit["n_labeled_instances"], "groups": audit["leakage"]["n_groups_labeled"],
                    "per_class": audit["instances_per_class"], "groups_per_class": audit["independent_groups_per_class"],
                    "no_annotation": audit["n_images_without_annotation"], "orphans": audit["n_orphan_annotations"]},
        "runtime": {k: v["mean_ms"] for k, v in rt["stages"].items()} | {
            "fps": rt["throughput_fps"], "ort": rt["embedder"]["onnxruntime_ms"]["mean_ms"], "torch": rt["embedder"]["pytorch_ms"]["mean_ms"],
            "cpu": rt["hardware"]["cpu"], "cores": rt["hardware"]["logical_cores"]},
        "pose_variants": {k: {"rate": v["detection_rate"], "ms": v["latency"]["mean_ms"], "sujud": v["detection_rate_by_class"]["sujud"]}
                          for k, v in pv["variants"].items()},
        "temporal": {mode: {d: {k: tb["results"][mode][d][k]["mean"] for k in ("phase_acc", "edit", "f1_50", "seg_ratio")}
                            for d in ("raw", "smoothing", "hmm", "hmm_local")} for mode in ("segment", "iid")},
        "demo": {"duration": demo["performance"]["frames_analyzed"] / 5.0, "plan": demo["plan"],
                 "raw": [{k: s[k] for k in ("start", "end", "phase", "uncertain")} for s in demo["raw"]],
                 "decoded": [{k: s[k] for k in ("start", "end", "phase", "uncertain", "confidence")} for s in demo["decoded"]],
                 "fps": demo["performance"]["analysis_fps"]},
        "localization": {"detector": int((loc.loc_source == "detector").sum()), "pose": int((loc.loc_source == "pose").sum()),
                         "full": int((loc.loc_source == "full_image").sum()), "iou50": float((loc.loc_best_iou >= 0.5).mean())},
        # counts of tests that pass in docs/test_report.md (update after re-running the suites)
        "tests": {"python": 55, "frontend_unit": 12, "e2e": 5},
    }
    out = ROOT / "portfolio" / "assets" / "data" / "deck_data.json"
    out.write_text(json.dumps(deck, indent=1), encoding="utf-8")
    print("wrote", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
