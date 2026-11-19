#!/usr/bin/env python
"""Render evaluation figures from saved artefacts (never from hand-typed numbers).

Inputs : artifacts/metrics/*.json|csv, artifacts/demo/demo_analysis.json (optional)
Outputs: artifacts/figures/*.png|svg, and copies + source data in portfolio/assets/
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from salat_vision.config import get_config  # noqa: E402
from salat_vision.labels import PHASE_LABELS_EN, PHASES, VISUAL_CLASSES, VISUAL_LABELS_EN  # noqa: E402
from salat_vision.viz import (AXIS, GRID, INK, INK_2, INK_MUTED, NEUTRAL, PHASE_COLORS, SERIES, SEQ_BLUE,  # noqa: E402
                              SURFACE, apply_style, hairline_grid, rounded_bars, save)

MODEL_NAMES = {
    "baseline_majority": "Majority-class baseline",
    "pose_rules": "Pose rules (no training)",
    "pose_logreg": "Pose features + logistic reg.",
    "pose_svm_rbf": "Pose features + RBF SVM",
    "pose_random_forest": "Pose features + random forest",
    "pose_mlp": "Pose features + MLP",
    "cnn_probe_mobilenet_v3_large": "MobileNetV3 frozen + logistic reg.",
    "cnn_probe_resnet18": "ResNet18 frozen + logistic reg.",
    "cnn_ft_mobilenet_v3_large": "MobileNetV3 fine-tuned",
    "cnn_ft_resnet18": "ResNet18 fine-tuned",
    "fusion_pose_cnn": "Fusion: pose + MobileNetV3 (selected)",
}
SHORT = {"qiyam": "Standing", "ruku": "Bowing", "sujud": "Prostration", "julus": "Sitting"}


def fig_model_comparison(cmp: pd.DataFrame, selected: str, out: Path) -> Path:
    apply_style()
    d = cmp.sort_values("crop_macro_f1", ascending=True).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, gridspec_kw={"wspace": 0.14})
    for ax, view, title in ((axes[0], "crop", "Annotated person crops"), (axes[1], "image", "Whole images (automatic localisation)")):
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, len(d) - 0.4)
        ys = np.arange(len(d))
        cols = [SERIES[0] if m == selected else "#c9ccd1" for m in d["model"]]
        rounded_bars(ax, ys, d[f"{view}_macro_f1"].values, color=cols, width=0.62, horizontal=True)
        lo, hi, mu = d[f"{view}_macro_f1_ci_lo"].values, d[f"{view}_macro_f1_ci_hi"].values, d[f"{view}_macro_f1"].values
        ax.hlines(ys, lo, hi, color=INK, linewidth=1, zorder=4)
        ax.vlines(lo, ys - 0.13, ys + 0.13, color=INK, linewidth=1, zorder=4)
        ax.vlines(hi, ys - 0.13, ys + 0.13, color=INK, linewidth=1, zorder=4)
        k = int(np.flatnonzero(d["model"].values == selected)[0])
        ax.text(hi[k] + 0.02, k, f"{mu[k]:.2f}", va="center", fontsize=10, color=INK, fontweight="semibold")
        hairline_grid(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Macro-F1 (mean of 3 repeats, 95% cluster-bootstrap CI)")
    axes[0].set_yticks(np.arange(len(d)), [MODEL_NAMES.get(m, m) for m in d["model"]])
    fig.subplots_adjust(top=0.86)
    fig.suptitle("Model comparison under repeated stratified group 5-fold CV (40 instances, 29 groups)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="semibold", color=INK, y=0.99)
    return save(fig, out, also_svg=True)


def fig_confusion(cm_crop: list, cm_img: list, out: Path) -> Path:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"wspace": 0.35})
    for ax, cm, title in ((axes[0], cm_crop, "Annotated person crops"), (axes[1], cm_img, "Whole images")):
        m = np.array(cm)[:, :4].astype(float)
        rows = m.sum(axis=1, keepdims=True)
        frac = np.divide(m, rows, out=np.zeros_like(m), where=rows > 0)
        ax.imshow(frac, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{int(m[i, j])}\n{frac[i, j]:.0%}", ha="center", va="center", fontsize=9,
                        color="#ffffff" if frac[i, j] > 0.55 else INK, linespacing=1.3)
        labels = [SHORT[c] for c in VISUAL_CLASSES]
        ax.set_xticks(range(4), labels)
        ax.set_yticks(range(4), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.subplots_adjust(top=0.80)
    fig.suptitle("Selected model: out-of-fold confusion (counts over 3 repeats; colour = row share)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold", color=INK, y=0.99)
    return save(fig, out, also_svg=True)


def fig_training_curves(hist: dict, out: Path) -> Path:
    apply_style()
    keys = [k for k in ("cnn_ft_mobilenet_v3_large", "cnn_ft_resnet18") if k in hist]
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 3.8), sharey=True, gridspec_kw={"wspace": 0.08})
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        runs = hist[k]
        L = max(len(r["history"]) for r in runs)
        for metric, color, lab in (("train_loss", SERIES[1], "Training loss"), ("val_loss", SERIES[0], "Validation loss")):
            mat = np.full((len(runs), L), np.nan)
            for i, r in enumerate(runs):
                vals = [h.get(metric, np.nan) for h in r["history"]]
                mat[i, :len(vals)] = vals
                ax.plot(np.arange(1, len(vals) + 1), vals, color=color, alpha=0.12, linewidth=1)
            mean = np.nanmean(mat, axis=0)
            n = np.sum(~np.isnan(mat), axis=0)
            ok = n >= 3
            ax.plot(np.arange(1, L + 1)[ok], mean[ok], color=color, linewidth=2, label=lab)
        be = [r["best_epoch"] for r in runs]
        ax.set_title(f"{'MobileNetV3-Large' if 'mobilenet' in k else 'ResNet18'} — best epoch median {int(np.median(be))}", fontsize=11)
        ax.set_xlabel("Epoch")
        hairline_grid(ax, "y")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend(loc="upper right")
    fig.subplots_adjust(top=0.78)
    fig.suptitle("Fine-tuning curves across 15 folds (thin = per fold, thick = mean while >=3 folds still training)",
                 x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK, y=0.99)
    return save(fig, out, also_svg=True)


def fig_calibration(cv: dict, out: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, zorder=1)
    for k, color, name in (("fusion_pose_cnn", SERIES[0], "Fusion (selected)"), ("pose_logreg", SERIES[1], "Pose + logistic reg."),
                           ("cnn_probe_mobilenet_v3_large", SERIES[2], "MobileNetV3 probe")):
        cal = cv["models"][k]["crop"]["calibration"]
        b = [x for x in cal["bins"] if x["n"]]
        ax.plot([x["confidence"] for x in b], [x["accuracy"] for x in b], color=color, linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2, label=f"{name} (ECE {cal['ece']:.2f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Mean top-class score in bin")
    ax.set_ylabel("Observed accuracy in bin")
    hairline_grid(ax, "both")
    ax.legend(loc="lower right")
    ax.set_title("Reliability: scores are under-confident")
    return save(fig, out, also_svg=True)


def fig_pose_variants(pb: dict, out: Path) -> Path:
    apply_style()
    variants = ["lite", "full", "heavy"]
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    x = np.arange(len(VISUAL_CLASSES))
    w = 0.24
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, len(x) - 0.4)
    for i, v in enumerate(variants):
        vals = [pb["variants"][v]["detection_rate_by_class"][c] for c in VISUAL_CLASSES]
        rounded_bars(ax, x + (i - 1) * (w + 0.02), vals, color=SERIES[i], width=w)
    ax.set_xticks(x, [SHORT[c] for c in VISUAL_CLASSES])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    hairline_grid(ax, "y")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[i]) for i in range(3)]
    ax.legend(handles, [f"{v} — {pb['variants'][v]['detection_rate']:.0%} overall, {pb['variants'][v]['latency']['mean_ms']:.0f} ms"
                        for v in variants], loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12), fontsize=8.5)
    ax.set_ylabel("Pose found on annotated crop")
    ax.set_title("MediaPipe pose variants: prostration is the hard case")
    return save(fig, out, also_svg=True)


def fig_runtime(rt: dict, out: Path) -> Path:
    apply_style()
    names = {"localize": "Person localisation", "pose": "Pose landmarks", "pose_classifier": "Pose classifier", "image_model": "Image model (ONNX)"}
    st = [(names[k], rt["stages"][k]["mean_ms"], rt["stages"][k]["p95_ms"]) for k in names if k in rt["stages"]]
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    ys = np.arange(len(st))
    mx = max(s[2] for s in st)
    ax.set_xlim(0, mx * 1.25)
    ax.set_ylim(-0.6, len(st) - 0.4)
    ax.invert_yaxis()
    rounded_bars(ax, ys, [s[1] for s in st], color=SERIES[0], width=0.55, horizontal=True)
    for y, (_, m, p) in zip(ys, st):
        ax.text(m + mx * 0.02, y, f"{m:.1f} ms (p95 {p:.0f})", va="center", fontsize=9, color=INK_2)
    ax.set_yticks(ys, [s[0] for s in st])
    ax.spines["left"].set_visible(False)
    hairline_grid(ax, "x")
    ax.set_xlabel("Mean latency per frame (ms)")
    tot = rt["stages"]["total"]["mean_ms"]
    ax.set_title(f"CPU latency per frame: {tot:.0f} ms total, {rt['throughput_fps']:.1f} frames/s ({rt['hardware']['logical_cores']} vCPU)")
    return save(fig, out, also_svg=True)


def fig_temporal(tb: dict, out: Path) -> Path:
    apply_style()
    decs = ["raw", "smoothing", "hmm", "hmm_local"]
    dn = {"raw": "Raw per-frame", "smoothing": "Smoothing + rules", "hmm": "Global HMM", "hmm_local": "HMM + local rules (used)"}
    metrics = [("phase_acc", "Frame phase accuracy", 100), ("edit", "Segmental edit score", 1), ("f1_50", "Segment F1@50", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (m, title, scale) in zip(axes, metrics):
        ax.set_ylim(-0.7, len(decs) - 0.3)
        ax.set_xlim(0, 105)
        ax.invert_yaxis()
        for j, mode in enumerate(("segment", "iid")):
            vals = [tb["results"][mode][d][m]["mean"] * scale for d in decs]
            ys = np.arange(len(decs)) + (j - 0.5) * 0.36
            rounded_bars(ax, ys, vals, color=SERIES[j], width=0.32, horizontal=True)
            for y, v in zip(ys, vals):
                ax.text(v + 1.2, y, f"{v:.0f}", va="center", fontsize=8.5, color=INK_2)
        ax.set_title(title, fontsize=11)
        hairline_grid(ax, "x")
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(np.arange(len(decs)), [dn[d] for d in decs])
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[j]) for j in range(2)]
    fig.legend(handles, ["Segment-consistent errors (conservative)", "Independent per-frame errors (optimistic)"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.subplots_adjust(top=0.78)
    fig.suptitle(f"Temporal decoders on {tb['results']['segment']['n_sequences']} semi-synthetic 2-4 rakaat sequences per regime "
                 "(emissions = real out-of-fold scores)", x=0.01, ha="left", fontsize=12, fontweight="semibold", color=INK, y=0.99)
    return save(fig, out, also_svg=True)


def fig_timeline(demo: dict, truth: dict | None, out: Path) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(12, 2.7))
    total = demo["video"]["duration_seconds"]
    rows = []
    if truth:
        rows.append(("Clip plan (ground truth)", [{"start": t["start"], "end": t["end"], "phase": t["phase"], "uncertain": False}
                                                  for t in truth["timeline"]]))
    rows.append(("Raw per-frame argmax", demo["raw_segments"]))
    rows.append(("Decoded (HMM + local rules)", demo["segments"]))
    for r, (name, segs) in enumerate(rows):
        for s in segs:
            col = NEUTRAL if s.get("uncertain") else PHASE_COLORS[s["phase"]]
            ax.add_patch(plt.Rectangle((s["start"], r - 0.32), max(s["end"] - s["start"] - 0.06, 0.02), 0.64,
                                       facecolor=col, edgecolor="none"))
    ax.set_xlim(0, total)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Time (s)")
    ax.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[p]) for p in PHASES] + [plt.Rectangle((0, 0), 1, 1, color=NEUTRAL)]
    ax.legend(handles, [PHASE_LABELS_EN[p] for p in PHASES] + ["Uncertain (score < 0.55)"], loc="upper center",
              bbox_to_anchor=(0.5, -0.32), ncol=4, fontsize=8.5)
    n_raw, n_dec = len(demo["raw_segments"]), len(demo["segments"])
    ax.set_title(f"Synthetic demo clip (seen images — mechanics, not accuracy): {n_raw} raw segments -> {n_dec} decoded")
    return save(fig, out, also_svg=True)


def fig_sample_grid(cfg, out: Path) -> Path | None:
    """Anonymised (blurred faces) examples per class with the detected pose skeleton."""
    import cv2

    from salat_vision.data.dataset import build_index
    from salat_vision.data.voc import load_image_rgb
    from salat_vision.features.pose import POSE_EDGES, crop_with_padding
    from salat_vision.privacy import Anonymizer

    idx = build_index(cfg=cfg)
    li = idx.labeled_instances
    picks = {"qiyam": ["ck2upgs2ttpwm07575jjfctev", "qiam(94)", "ck35sqichhe0n0757m66gwoty"],
             "ruku": ["ck2ygl8aj78s5072595cehoma", "ruku109", "ck2oth5dc5xxk07570b711dda"],
             "sujud": ["ck2osjowcmuaf07252lbqno9g", "ck34yexrazmie0757n3kns0l8", "ck2oufih0wy2m0794frm9jrhd"],
             "julus": ["IMG-20191019-WA0062", "SittingMO56", "ck2or3u694ynn0757ptzgbv8t"]}
    anon = Anonymizer(cfg.path("pretrained_dir"))
    tiles = []
    for c in VISUAL_CLASSES:
        row = []
        for image_id in picks[c]:
            r = li[li["image_id"] == image_id].iloc[0]
            img = load_image_rgb(idx.dataset_dir / r.file)
            box = (r.xmin, r.ymin, r.xmax, r.ymax)
            blurred, _ = anon(img, [box])
            crop, cb = crop_with_padding(blurred, box, 0.15)
            pose = anon.pose.detect(crop_with_padding(img, box, 0.15)[0])
            crop = crop.copy()
            if pose.detected:
                h, w = crop.shape[:2]
                lm = pose.landmarks
                t = max(2, int(max(h, w) / 160))
                for a, b in POSE_EDGES:
                    if a < 11 or b < 11:
                        continue
                    if lm[a, 3] > 0.35 and lm[b, 3] > 0.35:
                        cv2.line(crop, (int(lm[a, 0] * w), int(lm[a, 1] * h)), (int(lm[b, 0] * w), int(lm[b, 1] * h)),
                                 (255, 255, 255), t + 1, cv2.LINE_AA)
            h, w = crop.shape[:2]
            s = 260 / max(h, w)
            crop = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
            tile = np.full((270, 270, 3), 246, np.uint8)
            y0, x0 = (270 - crop.shape[0]) // 2, (270 - crop.shape[1]) // 2
            tile[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
            row.append(tile)
        tiles.append(row)
    anon.close()
    apply_style()
    fig, axes = plt.subplots(3, 4, figsize=(10, 8.2), gridspec_kw={"wspace": 0.04, "hspace": 0.04})
    for j, c in enumerate(VISUAL_CLASSES):
        for i in range(3):
            ax = axes[i, j]
            ax.imshow(tiles[j][i])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        axes[0, j].set_title(VISUAL_LABELS_EN[c], fontsize=11)
    fig.suptitle("Dataset examples per class with MediaPipe skeleton (faces blurred)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="semibold", color=INK)
    return save(fig, out)


def main() -> int:
    cfg = get_config()
    m = cfg.path("artifacts_dir") / "metrics"
    figs = cfg.path("artifacts_dir") / "figures"
    port = cfg.root / "portfolio" / "assets"
    data_dir = port / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cmp = pd.read_csv(m / "model_comparison.csv")
    cv = json.loads((m / "cv_results.json").read_text())
    sel = cfg["final_model"]["cv_model_key"]
    made = {}
    made["model_comparison"] = fig_model_comparison(cmp, sel, figs / "model_comparison.png")
    e = cv["models"][sel]
    made["confusion_matrix"] = fig_confusion(e["crop"]["pooled"]["confusion_matrix"], e["image"]["pooled"]["confusion_matrix"],
                                             figs / "confusion_matrix.png")
    made["training_curves"] = fig_training_curves(json.loads((m / "training_histories.json").read_text()), figs / "training_curves.png")
    made["calibration"] = fig_calibration(cv, figs / "calibration_reliability.png")
    pb = json.loads((m / "pose_variant_benchmark.json").read_text())
    made["pose_variants"] = fig_pose_variants(pb, figs / "pose_variant_detection.png")
    rt = json.loads((m / "runtime_benchmark.json").read_text())
    made["benchmark_chart"] = fig_runtime(rt, figs / "benchmark_chart.png")
    tb = json.loads((m / "temporal_benchmark.json").read_text())
    made["temporal_benchmark"] = fig_temporal(tb, figs / "temporal_benchmark.png")
    demo_p = cfg.path("artifacts_dir") / "demo" / "demo_analysis.json"
    if demo_p.is_file():
        demo = json.loads(demo_p.read_text())
        tp = cfg.path("artifacts_dir") / "demo" / "demo_sequence.json"
        truth = json.loads(tp.read_text()) if tp.is_file() else None
        made["posture_timeline"] = fig_timeline(demo, truth, figs / "posture_timeline.png")
        (data_dir / "posture_timeline_segments.json").write_text(json.dumps(
            {"decoded": demo["segments"], "raw": demo["raw_segments"], "plan": truth["timeline"] if truth else None,
             "performance": demo["performance"]}, indent=1), encoding="utf-8")
    try:
        made["dataset_samples"] = fig_sample_grid(cfg, figs / "dataset_samples_anonymized.png")
    except Exception as exc:  # dataset not extracted
        print("sample grid skipped:", exc)

    # portfolio copies + source data
    names = {"model_comparison": "model_comparison.png", "confusion_matrix": "confusion_matrix.png",
             "training_curves": "training_curves.png", "calibration": "calibration_reliability.png",
             "pose_variants": "pose_variant_detection.png", "benchmark_chart": "benchmark_chart.png",
             "temporal_benchmark": "temporal_benchmark.png", "posture_timeline": "posture_timeline.png",
             "dataset_samples": "dataset_samples_anonymized.png"}
    for k, p in made.items():
        if p:
            shutil.copy(p, port / names[k])
    shutil.copy(figs / "dataset_class_distribution.png", port / "dataset_class_distribution.png")
    shutil.copy(figs / "dataset_source_by_class.png", port / "dataset_source_by_class.png")
    shutil.copy(figs / "dataset_box_geometry.png", port / "dataset_box_geometry.png")
    cmp.to_csv(data_dir / "model_comparison.csv", index=False)
    pd.DataFrame(e["crop"]["pooled"]["confusion_matrix"], index=VISUAL_CLASSES,
                 columns=VISUAL_CLASSES + ["abstain"]).to_csv(data_dir / "confusion_matrix_crop.csv")
    pd.DataFrame(e["image"]["pooled"]["confusion_matrix"], index=VISUAL_CLASSES,
                 columns=VISUAL_CLASSES + ["abstain"]).to_csv(data_dir / "confusion_matrix_image.csv")
    audit = json.loads((m / "dataset_audit.json").read_text())
    pd.DataFrame({"class": VISUAL_CLASSES, "instances": [audit["instances_per_class"][c] for c in VISUAL_CLASSES],
                  "images": [audit["images_per_class"][c] for c in VISUAL_CLASSES],
                  "independent_groups": [audit["independent_groups_per_class"][c] for c in VISUAL_CLASSES]}
                 ).to_csv(data_dir / "dataset_class_distribution.csv", index=False)
    for f in ("runtime_benchmark.json", "pose_variant_benchmark.json", "temporal_benchmark.json", "training_histories.json"):
        shutil.copy(m / f, data_dir / f)
    pd.DataFrame([{"model": k, **{f"{c}_{s}": e2["crop"]["pooled"]["per_class"][c][s] for c in VISUAL_CLASSES for s in ("precision", "recall", "f1")}}
                  for k, e2 in cv["models"].items()]).to_csv(data_dir / "per_class_metrics_crop.csv", index=False)
    print("figures:", ", ".join(sorted(p.name for p in made.values() if p)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
