#!/usr/bin/env python
"""Generate the four project notebooks (and optionally execute them in place).

    python scripts/build_notebooks.py            # write notebooks/*.ipynb
    python scripts/build_notebooks.py --execute  # write + execute with real outputs

The notebooks call the same package functions as the scripts, so there is no hidden
notebook-only logic; they only orchestrate, visualise and explain.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
import sys, os, json, warnings
from pathlib import Path
ROOT = Path.cwd().resolve()
while not (ROOT / "configs" / "default.yaml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("GLOG_minloglevel", "2")
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
pd.set_option("display.max_columns", 30); pd.set_option("display.width", 160)
from salat_vision.config import get_config
from salat_vision.viz import apply_style
apply_style()
cfg = get_config()
print("project root:", ROOT)"""


def md(s: str):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str):
    return nbf.v4.new_code_cell(s.strip("\n"))


def nb(cells) -> nbf.NotebookNode:
    n = nbf.v4.new_notebook()
    n.cells = cells
    n.metadata["kernelspec"] = {"name": "salat-vision", "display_name": "Python (salat-vision)", "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    return n


# ----------------------------------------------------------------------------------------- 01
NB01 = nb([
    md("""
# 01 · Dataset audit — Salat Postures (provided sample)

This notebook locates the dataset (folder **or** ZIP), extracts it safely, and audits what is actually inside:
structure, labels, quality, near-duplicates and leakage risks. Every number below is computed from the files.

Paths come from `configs/default.yaml` (override with `SALAT_DATASET_ZIP` / `SALAT_DATASET_DIR`)."""),
    code(SETUP),
    md("## 1. Discover and (safely) extract the dataset"),
    code("""\
from salat_vision.data.extract import discover_dataset, inspect_zip, safe_extract, DATASET_NAME
dataset_dir = cfg.path("dataset_dir")
zip_path = os.environ.get("SALAT_DATASET_ZIP") or cfg.get("paths", "dataset_zip")
print("configured dataset folder:", dataset_dir.relative_to(ROOT), "| exists:", dataset_dir.is_dir())
if zip_path:
    rep = inspect_zip(zip_path)
    print(f"archive: {rep.n_files} files, {rep.uncompressed_bytes/1e6:.2f} MB, safe={rep.is_safe}, prefix='{rep.common_prefix}'")
    safe_extract(zip_path, cfg.path("raw_dir"))
elif not dataset_dir.is_dir():
    print("discovery:", discover_dataset())
extraction = json.loads((ROOT / "artifacts/reports/extraction_report.json").read_text())
print("recorded extraction:", {k: extraction[k] for k in ("n_entries", "n_files", "uncompressed_bytes", "is_safe", "extracted_to")})"""),
    code("""\
for p in sorted(dataset_dir.iterdir()):
    n = sum(1 for _ in p.rglob('*')) if p.is_dir() else None
    print(f"{p.name:40s}", f"{n} files" if n is not None else f"{p.stat().st_size} bytes")
print((dataset_dir / "README.md").read_text()[:900])"""),
    md("## 2. Image and annotation index"),
    code("""\
from salat_vision.data.dataset import build_index
idx = build_index(cfg=cfg)
im, inst = idx.images, idx.instances
print(f"image files: {len(im)} | with annotation: {im.has_annotation.sum()} | without: {(~im.has_annotation).sum()} | orphan XML: {len(idx.orphan_annotations)}")
print(f"annotated person instances: {len(idx.labeled_instances)} in {idx.labeled_instances.image_id.nunique()} images")
im[["image_id", "ext", "width", "height", "has_annotation", "n_objects", "labels", "source_kind"]].head(12)"""),
    code("""\
print("raw label spellings:", inst.raw_label.value_counts().to_dict())
print("box formats:", inst.box_format.value_counts().to_dict())
idx.orphan_annotations"""),
    md("## 3. Class distribution — only four postures are annotated"),
    code("""\
from salat_vision.data import audit
idx2, summary = audit.run_audit(cfg)
li = idx2.labeled_instances
display(pd.DataFrame({"instances": summary["instances_per_class"], "images": summary["images_per_class"],
                      "independent groups": summary["independent_groups_per_class"]}))
display(Image(str(ROOT / "artifacts/figures/dataset_class_distribution.png"), width=720))
pd.Series(summary["phase_support"], name="support in this dataset").to_frame()"""),
    md("## 4. Quality checks"),
    code("""\
print("unreadable images:", summary["n_unreadable_images"], "| EXIF-rotated:", summary["exif_rotated_images"],
      "| XML size mismatches:", summary["xml_size_mismatches"], "| exact duplicates:", summary["exact_duplicate_files"])
print("width px:", summary["width_px"], "\\nheight px:", summary["height_px"])
print("box area / image area:", {k: round(v, 3) for k, v in summary["box_rel_area"].items()})
pd.DataFrame(summary["issues"])"""),
    code("""display(Image(str(ROOT / "artifacts/figures/dataset_resolution.png"), width=560))
display(Image(str(ROOT / "artifacts/figures/dataset_box_geometry.png"), width=720))"""),
    md("## 5. Leakage: near-duplicates and capture sessions"),
    code("""\
links = pd.DataFrame(summary["leakage"]["group_links"])
print(f"labelled groups: {summary['leakage']['n_groups_labeled']} | multi-instance groups: {summary['leakage']['n_multi_instance_groups']} | largest group: {summary['leakage']['largest_group_instances']} instances")
links"""),
    code("""\
splits = json.loads((ROOT / "data/splits/cv_splits.json").read_text())
groups = pd.read_csv(ROOT / "data/splits/instance_groups.csv").set_index("instance_id")
rows = []
for f in splits["folds"][:5]:
    tr, te = set(groups.loc[f["train"], "group_id"]), set(groups.loc[f["test"], "group_id"])
    rows.append({"fold": f["fold"], "train": len(f["train"]), "test": len(f["test"]), "shared groups": len(tr & te),
                 **groups.loc[f["test"], "label"].value_counts().to_dict()})
pd.DataFrame(rows).fillna(0)"""),
    md("## 6. Shortcut risk: capture source vs. class"),
    code("""display(Image(str(ROOT / "artifacts/figures/dataset_source_by_class.png"), width=720))"""),
    md("## 7. Examples (faces blurred) with the detected pose skeleton"),
    code("""\
p = ROOT / "artifacts/figures/dataset_samples_anonymized.png"
display(Image(str(p), width=760)) if p.exists() else print("run scripts/make_figures.py to render the anonymised grid")"""),
    md("""
## 8. Findings and modelling recommendations

* The archive is the **public sample** of the Salat Postures Dataset: 41 images, 40 annotated people, 4 classes.
  The full release (1,904 images) is referenced in the README but is not part of the provided files.
* Two annotation files have no image, four images have no annotation, one annotation is stored as a polygon.
* Images are heterogeneous (web images, phone photos, video frames) and **capture source is confounded with class**.
* Near-duplicates and same-session frames exist across *different file-name families*, so random splits would leak.
  All evaluation therefore uses **stratified group k-fold** on 29 independent groups.
* Iktidal, the sitting between prostrations and tasyahud are **not annotated**; they can only be inferred from temporal
  order in a video. The models learn 4 visual postures; the 6 phases come from a temporal decoder.
* With ~29 effective samples, prefer strong priors (pretrained pose estimation, frozen ImageNet features) and report
  confidence intervals rather than point estimates."""),
])

# ----------------------------------------------------------------------------------------- 02
NB02 = nb([
    md("""
# 02 · Training the candidate models

Feature extraction, the leakage-aware cross-validation protocol and every candidate model, run from the same
package code as `scripts/run_experiments.py`. Cheap models are trained live here; the full 30-run CNN fine-tuning sweep
(~10 min on 4 CPU cores) is loaded from `artifacts/metrics` and one fold is re-trained live to show the loop."""),
    code(SETUP),
    code("""\
import platform, torch, torchvision, mediapipe, sklearn, onnxruntime
print("python", platform.python_version(), "| torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| mediapipe", mediapipe.__version__, "| scikit-learn", sklearn.__version__, "| onnxruntime", onnxruntime.__version__)
print("cpu cores:", os.cpu_count(), "| cuda:", torch.cuda.is_available())
cfg["splits"], cfg["cnn"]"""),
    md("## 1. Features: pose landmarks and CNN embeddings"),
    code("""\
from salat_vision.experiments import build_feature_cache
from salat_vision.features.pose_features import FEATURE_NAMES
cache = build_feature_cache(cfg)
X = pd.DataFrame(cache.X_pose, columns=FEATURE_NAMES)
X.insert(0, "label", cache.instances.label.values)
print("pose detected per class:", X.groupby("label").pose_detected.mean().round(2).to_dict())
print("embedding shapes:", {k: v["orig"].shape for k, v in cache.emb.items()})
X.head(8).round(2)"""),
    code("""\
from salat_vision.viz import VISUAL_COLORS
from salat_vision.labels import VISUAL_CLASSES, VISUAL_LABELS_EN
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for c in VISUAL_CLASSES:
    d = X[(X.label == c) & (X.pose_detected == 1)]
    ax.scatter(d.knee_angle_3d, d.torso_incl_3d, s=60, color=VISUAL_COLORS[c], edgecolor="#fcfcfb", linewidth=2, label=VISUAL_LABELS_EN[c])
ax.axvline(140, color="#c3c2b7", lw=1); ax.axhline(45, color="#c3c2b7", lw=1)
ax.set_xlabel("knee angle, 3-D (deg)  — 180 = straight leg"); ax.set_ylabel("torso inclination, 3-D (deg) — 0 = upright")
ax.legend(loc="upper left", fontsize=8); ax.set_title("Two pose features already separate most postures"); plt.show()"""),
    md("## 2. Protocol: repeated stratified group k-fold"),
    code("""\
from salat_vision.data.splits import make_cv_splits
sp = make_cv_splits(cache.y, cache.groups, n_splits=cfg["splits"]["n_splits"], n_repeats=cfg["splits"]["n_repeats"], seed=cfg.seed)
pd.DataFrame([{"repeat": s["repeat"], "fold": s["fold"], "n_train": len(s["train"]), "n_test": len(s["test"]),
               "test classes": dict(pd.Series(cache.instances.label.values[s["test"]]).value_counts())} for s in sp]).head(6)"""),
    md("## 3. Train and evaluate the fast models live (pose classifiers, CNN probes, fusion)"),
    code("""\
import time
from salat_vision.experiments import run_cv, summarize
fast = ["baseline_majority", "pose_rules", "pose_logreg", "pose_svm_rbf", "pose_random_forest", "pose_mlp",
        "cnn_probe_mobilenet_v3_large", "cnn_probe_resnet18", "fusion_pose_cnn"]
t0 = time.perf_counter()
cv = run_cv(cache, cfg, models=fast, log=lambda *a: None)
summ = summarize(cache, cv, n_boot=500, seed=cfg.seed)
print(f"cross-validation of {len(fast)} models: {time.perf_counter()-t0:.0f} s")
pd.DataFrame({m: {"macro-F1 (crops)": e["crop"]["macro_f1_mean"], "sd": e["crop"]["macro_f1_std"],
                  "macro-F1 (images)": e["image"]["macro_f1_mean"], "accuracy (crops)": e["crop"]["accuracy_mean"]}
              for m, e in summ.items()}).T.round(3).sort_values("macro-F1 (crops)", ascending=False)"""),
    md("## 4. CNN fine-tuning: one fold live, all 30 runs from the saved sweep"),
    code("""\
from salat_vision.data.splits import inner_train_val
from salat_vision.models.cnn import finetune, predict_proba_torch
s = sp[0]
tr, te = np.array(s["train"]), np.array(s["test"])
itr, iva = inner_train_val(tr, cache.y, cache.groups, val_fraction=0.2, seed=cfg.seed)
ckpt = ROOT / "models/checkpoints/notebook_demo_mobilenet_v3_large.pt"
res = finetune("mobilenet_v3_large", [cache.crops[i] for i in itr], cache.y[itr], [cache.crops[i] for i in iva], cache.y[iva],
               n_classes=4, cfg=cfg, seed=cfg.seed, checkpoint_path=ckpt)
pred = predict_proba_torch(res.model, [cache.crops[i] for i in te]).argmax(1)
print(f"trained {len(res.history)} epochs in {res.train_seconds:.0f}s, best epoch {res.best_epoch}, checkpoint {ckpt.name} ({ckpt.stat().st_size/1e6:.1f} MB)")
print("held-out fold accuracy:", (pred == cache.y[te]).mean().round(3), f"on {len(te)} people")
h = pd.DataFrame(res.history)
ax = h.plot(x="epoch", y=["train_loss", "val_loss"], figsize=(6, 3.2), color=["#eb6834", "#2a78d6"], title="This fold"); plt.show()"""),
    code("""display(Image(str(ROOT / "artifacts/figures/training_curves.png"), width=900))
full = pd.read_csv(ROOT / "artifacts/metrics/model_comparison.csv").set_index("model")
full.loc[["cnn_ft_mobilenet_v3_large", "cnn_ft_resnet18", "cnn_probe_mobilenet_v3_large"], ["crop_macro_f1", "crop_macro_f1_sd", "image_macro_f1", "train_seconds_total"]].round(3)"""),
    md("""
Fine-tuning drives the training loss to ~0.25 while validation loss stays near 1.0: with ~32 training people per fold the
network memorises. It is ~100x slower than the frozen probe and not better on held-out groups, so it is not deployed."""),
    md("## 5. Final model: train on all data and export (to a temporary folder, checked against the shipped bundle)"),
    code("""\
import tempfile
from salat_vision.config import load_config
from salat_vision.export import train_and_export
tmp = Path(tempfile.mkdtemp())
cfg_tmp = load_config(overrides={"paths": {"models_dir": str(tmp)}})
out = train_and_export(cfg_tmp, log=print)
new = json.loads((out / "manifest.json").read_text())
shipped = json.loads((ROOT / "models/salat_vision_v1/manifest.json").read_text())
cmp = pd.DataFrame([{"file": f["name"], "re-exported sha256": f["sha256"][:16], "shipped sha256": g["sha256"][:16],
                     "identical": f["sha256"] == g["sha256"]} for f, g in zip(new["files"], shipped["files"])])
cmp"""),
    md("""
The scikit-learn heads are bit-for-bit reproducible. If the ONNX bytes differ, it is only exporter metadata; the
embedding parity check (cosine ≥ 0.9999 against PyTorch on all crops) is enforced inside `train_and_export`."""),
])

# ----------------------------------------------------------------------------------------- 03
NB03 = nb([
    md("""
# 03 · Evaluation and model selection

All numbers come from `artifacts/metrics/` written by `scripts/run_experiments.py` (3 × stratified group 5-fold CV;
out-of-fold predictions; 95% CIs from a cluster bootstrap that resamples capture groups)."""),
    code(SETUP),
    code("""\
cvr = json.loads((ROOT / "artifacts/metrics/cv_results.json").read_text())
cmp = pd.read_csv(ROOT / "artifacts/metrics/model_comparison.csv")
cols = ["model", "family", "crop_macro_f1", "crop_macro_f1_ci_lo", "crop_macro_f1_ci_hi", "crop_accuracy",
        "image_macro_f1", "image_accuracy", "abstain_rate", "ece"]
cmp[cols].sort_values("crop_macro_f1", ascending=False).round(3)"""),
    code("""display(Image(str(ROOT / "artifacts/figures/model_comparison.png"), width=980))"""),
    md("## Selected model — per-class performance and confusions"),
    code("""\
sel = cfg["final_model"]["cv_model_key"]
e = cvr["models"][sel]
print(sel, "| crop macro-F1 %.3f [%.3f, %.3f]" % (e["crop"]["macro_f1_mean"], *e["crop"]["macro_f1_ci95"]),
      "| image macro-F1 %.3f [%.3f, %.3f]" % (e["image"]["macro_f1_mean"], *e["image"]["macro_f1_ci95"]))
display(pd.DataFrame(e["crop"]["pooled"]["per_class"]).T.round(3))
display(Image(str(ROOT / "artifacts/figures/confusion_matrix.png"), width=860))"""),
    md("## Calibration"),
    code("""\
display(Image(str(ROOT / "artifacts/figures/calibration_reliability.png"), width=480))
pd.DataFrame(e["crop"]["calibration"]["bins"])"""),
    md("""
The fused scores are **under-confident** (accuracy above the score in every populated bin). Averaging two models and
strong L2 regularisation both shrink scores. Consequence for the product: the "uncertain" flag (score < 0.55) is
conservative. Scores are shown as model confidence, never as calibrated probabilities."""),
    md("## Error analysis"),
    code("""\
oof = pd.read_csv(ROOT / "artifacts/metrics/oof_predictions.csv")
o = oof[(oof.model == sel) & (oof.view == "crop")]
err = o[o.true != o.pred].groupby(["id", "true", "pred"]).size().rename("times_wrong (of 3)").reset_index()
img = pd.read_csv(ROOT / "artifacts/metrics/dataset_images.csv").set_index("image_id")
err["source"] = err["id"].str.split("#").str[0].map(img["source_kind"])
err.sort_values("times_wrong (of 3)", ascending=False)"""),
    code("""\
from salat_vision.data.dataset import build_index
from salat_vision.data.voc import load_image_rgb
from salat_vision.features.pose import crop_with_padding
from salat_vision.privacy import Anonymizer
idx = build_index(cfg=cfg); li = idx.labeled_instances.set_index("instance_id")
anon = Anonymizer(cfg.path("pretrained_dir"))
worst = err.sort_values("times_wrong (of 3)", ascending=False).drop_duplicates("id").head(6)
fig, axes = plt.subplots(1, len(worst), figsize=(2.6 * len(worst), 3.2))
for ax, r in zip(np.atleast_1d(axes), worst.itertuples()):
    row = li.loc[r.id]; box = (row.xmin, row.ymin, row.xmax, row.ymax)
    blurred, _ = anon(load_image_rgb(idx.dataset_dir / row.file), [box])
    ax.imshow(crop_with_padding(blurred, box, 0.15)[0]); ax.set_axis_off()
    ax.set_title(f"true {r.true}\\npred {r.pred}", fontsize=9)
anon.close(); fig.suptitle("Most frequent errors (faces blurred)", x=0.01, ha="left"); plt.show()"""),
    md("## Does the model lean on capture source? (shortcut check)"),
    code("""\
o2 = o.copy(); o2["source"] = o2["id"].str.split("#").str[0].map(img["source_kind"])
o2["web"] = np.where(o2.source.str.startswith("web"), "web (Labelbox)", "own captures / video frames")
o2.assign(correct=o2.true == o2.pred).groupby(["web", "true"]).correct.agg(["mean", "size"]).round(2)"""),
    md("## Localisation (end-to-end view)"),
    code("""\
loc = pd.read_csv(ROOT / "artifacts/metrics/localization_eval.csv")
print(loc.loc_source.value_counts().to_dict(), "| IoU>=0.5 with an annotated box:", (loc.loc_best_iou >= 0.5).mean().round(3))
loc[loc.loc_best_iou < 0.5][["image_id", "label", "loc_source", "loc_best_iou", "pose_detected"]]"""),
    md("## Runtime"),
    code("""\
rt = json.loads((ROOT / "artifacts/metrics/runtime_benchmark.json").read_text())
pv = json.loads((ROOT / "artifacts/metrics/pose_variant_benchmark.json").read_text())
print(rt["hardware"]["cpu"], "|", rt["hardware"]["logical_cores"], "vCPU |", rt["throughput_fps"], "frames/s")
display(pd.DataFrame(rt["stages"]).T)
display(pd.DataFrame({v: {"detection rate": d["detection_rate"], "mean ms": d["latency"]["mean_ms"], "MB": d["size_mb"]} for v, d in pv["variants"].items()}).T)
display(Image(str(ROOT / "artifacts/figures/benchmark_chart.png"), width=640))"""),
    md("""
## Selection

* **Selected:** the pre-registered fusion of pose logistic regression and the frozen MobileNetV3 probe — highest mean
  macro-F1 on both views (0.80 crops, 0.68 images) and the lowest variance across repeats; chosen before results, so
  no selection bias; its components are fast (≈10 ms together after pose estimation).
* **Caveat:** confidence intervals of the top models overlap; with 29 independent groups the ranking is suggestive,
  not conclusive. The full dataset (and real prayer videos) are needed for a definitive comparison.
* **Rejected:** fine-tuned CNNs (overfit, ~100x slower to train, no gain), ResNet18 (worse than MobileNetV3 in every
  setting), training-free rules (useful baseline, but abstain when no pose is found)."""),
])

# ----------------------------------------------------------------------------------------- 04
NB04 = nb([
    md("""
# 04 · Video inference and temporal segmentation

The dataset has no prayer *videos*, so this demo uses a **synthetic clip** assembled from dataset images in prayer
order (faces blurred). The deployed model was trained on these images: the clip demonstrates the mechanics
(decoding, sampling, temporal decoding, annotation) — **it is not an accuracy measurement**. The quantitative
evaluation of the temporal decoder is the semi-synthetic benchmark at the end."""),
    code(SETUP),
    code("""\
demo = ROOT / "artifacts/demo/demo_sequence.mp4"
if not demo.exists():
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts/make_demo_video.py")], check=True)
plan = json.loads((ROOT / "artifacts/demo/demo_sequence.json").read_text())
print(f"clip: {plan['duration']} s, {plan['size']}, {len(plan['timeline'])} planned phases")
pd.DataFrame(plan["timeline"])[["start", "end", "phase"]].T"""),
    md("## Run the full pipeline (real model bundle)"),
    code("""\
from salat_vision.inference.predictor import PosturePredictor
from salat_vision.inference.video import analyze_video
out_video = ROOT / "artifacts/demo/demo_annotated_notebook.mp4"
with PosturePredictor(cfg=cfg) as pred:
    res = analyze_video(demo, pred, cfg, annotate_path=out_video)
print("decoder:", res["decoder"], "| frames analysed:", res["performance"]["frames_analyzed"],
      "| analysis fps:", res["performance"]["analysis_fps"], "| real-time factor:", res["performance"]["realtime_factor"])
pd.DataFrame(res["segments"])[["start", "end", "duration", "label", "confidence", "uncertain", "evidence"]]"""),
    code("""\
s = res["summary"]
print("observed sequence:", " -> ".join(s["sequence"]))
print("cycles observed:", s["cycles_observed"], "| uncertain segments:", s["n_uncertain_segments"], f"({s['uncertain_seconds']} s)")
pd.Series(s["phase_seconds"], name="seconds").to_frame().T"""),
    md("## Raw frame predictions vs. temporal decoding vs. the clip plan"),
    code("""\
from salat_vision.viz import PHASE_COLORS, NEUTRAL, VISUAL_COLORS
fig, ax = plt.subplots(figsize=(12, 2.4))
rows = [("plan", [{**t, "uncertain": False} for t in plan["timeline"]]), ("raw", res["raw_segments"]), ("decoded", res["segments"])]
for r, (name, segs) in enumerate(rows):
    for seg in segs:
        ax.add_patch(plt.Rectangle((seg["start"], r - 0.32), seg["end"] - seg["start"] - 0.06, 0.64,
                                   color=NEUTRAL if seg.get("uncertain") else PHASE_COLORS[seg["phase"]]))
ax.set_xlim(0, res["video"]["duration_seconds"]); ax.set_ylim(2.6, -0.6); ax.set_yticks(range(3), [r[0] for r in rows])
ax.set_xlabel("time (s)"); plt.show()
print(f"raw segments: {len(res['raw_segments'])} -> decoded: {len(res['segments'])}")
truth = [next(t['phase'] for t in plan['timeline'] if t['start'] <= f['t'] < t['end']) for f in res['frames']]
print("frame agreement with the plan (seen images):", np.mean([a == f['phase'] for a, f in zip(truth, res['frames'])]).round(3))"""),
    code("""\
fr = pd.DataFrame([{"t": f["t"], **(f["probabilities"] or {})} for f in res["frames"]])
fig, ax = plt.subplots(figsize=(12, 3))
for c in ["qiyam", "ruku", "sujud", "julus"]:
    ax.plot(fr.t, fr[c], color=VISUAL_COLORS[c], lw=1.6, label=c)
ax.axhline(cfg["inference"]["uncertainty_threshold"], color="#898781", lw=1)
ax.set_ylim(0, 1); ax.set_xlabel("time (s)"); ax.set_ylabel("frame score"); ax.legend(ncol=4, loc="upper right"); plt.show()"""),
    md("## Annotated frames"),
    code("""\
import imageio_ffmpeg
gen = imageio_ffmpeg.read_frames(str(out_video), output_params=["-vf", "fps=0.25"])
meta = next(gen); W, H = meta["size"]
frames = [np.frombuffer(b, np.uint8).reshape(H, W, 3) for b in gen]
fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
for ax, f in zip(axes, frames[1:5]):
    ax.imshow(f); ax.set_axis_off()
plt.show()"""),
    md("## Temporal decoder benchmark (semi-synthetic, real out-of-fold scores)"),
    code("""\
tb = json.loads((ROOT / "artifacts/metrics/temporal_benchmark.json").read_text())
rows = [{"regime": m, "decoder": d, **{k: tb["results"][m][d][k]["mean"] for k in ("phase_acc", "edit", "f1_50", "seg_ratio")}}
        for m in ("segment", "iid") for d in ("raw", "smoothing", "hmm", "hmm_local")]
display(pd.DataFrame(rows).round(3))
display(Image(str(ROOT / "artifacts/figures/temporal_benchmark.png"), width=980))"""),
    md("""
The global grammar HMM gives the cleanest segmentation, but when a whole segment is misrecognised it can desynchronise
the contextual names (iktidal vs. standing, tasyahud vs. sitting between prostrations). The deployed **hybrid** keeps
the HMM segmentation and names phases from local context — best on phase accuracy and F1@50 in both error regimes,
selected on a separate validation seed."""),
    code("""\
dst = ROOT / "artifacts/predictions/demo_analysis_notebook.json"
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({k: v for k, v in res.items() if k != "frames"}, indent=1))
print("saved", dst.relative_to(ROOT))"""),
])


NOISE = ("INFO: [", "WARNING: [", "WARNING: Logging before InitGoogle", "W0000", "I0000", "INFO: Created TensorFlow")


def strip_native_logs(book: nbf.NotebookNode) -> int:
    """Remove MediaPipe/LiteRT native log lines (written to fd 2) from stderr outputs."""
    removed = 0
    for cell in book.cells:
        if cell.cell_type != "code":
            continue
        keep = []
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and out.get("name") == "stderr":
                lines = out["text"].splitlines(keepends=True)
                clean = [ln for ln in lines if not ln.startswith(NOISE)]
                removed += len(lines) - len(clean)
                if not "".join(clean).strip():
                    continue
                out["text"] = "".join(clean)
            keep.append(out)
        cell["outputs"] = keep
    return removed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--clean-only", action="store_true", help="strip native log noise from existing notebooks")
    args = ap.parse_args()
    if args.clean_only:
        for path in sorted(NB_DIR.glob("*.ipynb")):
            book = nbf.read(path, as_version=4)
            n = strip_native_logs(book)
            nbf.write(book, path)
            print(f"{path.name}: removed {n} native log lines")
        return 0
    NB_DIR.mkdir(exist_ok=True)
    books = {"01_dataset_audit.ipynb": NB01, "02_train_models.ipynb": NB02,
             "03_evaluate_and_compare.ipynb": NB03, "04_video_inference_demo.ipynb": NB04}
    for name, book in books.items():
        if args.only and name not in args.only:
            continue
        path = NB_DIR / name
        if args.execute:
            from nbclient import NotebookClient

            t0 = time.perf_counter()
            NotebookClient(book, timeout=1800, kernel_name="salat-vision", resources={"metadata": {"path": str(NB_DIR)}}).execute()
            print(f"executed {name} in {time.perf_counter() - t0:.0f}s; removed {strip_native_logs(book)} native log lines")
        nbf.write(book, path)
        nbf.validate(nbf.read(path, as_version=4))
        print("wrote", path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
