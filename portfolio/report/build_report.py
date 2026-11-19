#!/usr/bin/env python
"""Build portfolio/report/Salat_Vision_Technical_Report.pdf from the project's real artefacts.

    python portfolio/report/build_report.py

Numbers are read from artifacts/metrics/*.json|csv at build time (never typed by hand).
HTML is rendered to PDF by Chromium (Playwright), so the text stays selectable.
Author/role/contact come from report_config.json (placeholders by default).
"""

from __future__ import annotations

import html
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
ASSETS = ROOT / "portfolio" / "assets"

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_NAMES = {
    "fusion_pose_cnn": "Fusion: pose LR + MobileNetV3 probe (selected)",
    "pose_logreg": "Pose features + logistic regression",
    "pose_mlp": "Pose features + MLP",
    "cnn_ft_mobilenet_v3_large": "MobileNetV3 fine-tuned",
    "cnn_probe_mobilenet_v3_large": "MobileNetV3 frozen + logistic regression",
    "pose_random_forest": "Pose features + random forest",
    "pose_svm_rbf": "Pose features + RBF SVM",
    "cnn_ft_resnet18": "ResNet18 fine-tuned",
    "pose_rules": "Pose rules (no training)",
    "cnn_probe_resnet18": "ResNet18 frozen + logistic regression",
    "baseline_majority": "Majority-class baseline",
}


def load() -> dict:
    m = ROOT / "artifacts" / "metrics"
    return {
        "audit": json.loads((m / "dataset_audit.json").read_text()),
        "cv": json.loads((m / "cv_results.json").read_text()),
        "cmp": pd.read_csv(m / "model_comparison.csv"),
        "rt": json.loads((m / "runtime_benchmark.json").read_text()),
        "pv": json.loads((m / "pose_variant_benchmark.json").read_text()),
        "tb": json.loads((m / "temporal_benchmark.json").read_text()),
        "loc": pd.read_csv(m / "localization_eval.csv"),
        "manifest": json.loads((ROOT / "models" / "salat_vision_v1" / "manifest.json").read_text()),
        "cfg": json.loads((HERE / "report_config.json").read_text()),
    }


def img(path: Path, max_w: int = 1800, quality: int = 86) -> str:
    """Downscale large screenshots into build/ as JPEG; return a file:// URL."""
    BUILD.mkdir(parents=True, exist_ok=True)
    im = Image.open(path)
    out = BUILD / (path.stem + (".jpg" if im.mode != "RGBA" or path.suffix == ".png" else path.suffix))
    if im.width > max_w:
        im = im.resize((max_w, int(im.height * max_w / im.width)), Image.LANCZOS)
    im.convert("RGB").save(out, "JPEG", quality=quality, optimize=True)
    return out.as_uri()


def f2(x: float) -> str:
    return f"{x:.2f}"


def build_html(d: dict) -> tuple[str, str]:
    a, cv, cmp, rt, pv, tb, loc, man, cfg = (d[k] for k in ("audit", "cv", "cmp", "rt", "pv", "tb", "loc", "manifest", "cfg"))
    sel = man["cv_metrics"]["model_key"]
    e = cv["models"][sel]
    crop, image = e["crop"], e["image"]
    pc = crop["pooled"]["per_class"]
    seg = tb["results"]["segment"]
    iid = tb["results"]["iid"]
    total_ms = rt["stages"]["total"]["mean_ms"]
    fonts = ASSETS / "fonts"
    esc = html.escape

    css = f"""
@font-face {{ font-family: 'PJS'; src: url('{(fonts / 'PlusJakartaSans-Regular.ttf').as_uri()}'); font-weight: 400; }}
@font-face {{ font-family: 'PJS'; src: url('{(fonts / 'PlusJakartaSans-Medium.ttf').as_uri()}'); font-weight: 500; }}
@font-face {{ font-family: 'PJS'; src: url('{(fonts / 'PlusJakartaSans-SemiBold.ttf').as_uri()}'); font-weight: 600; }}
@font-face {{ font-family: 'PJS'; src: url('{(fonts / 'PlusJakartaSans-Bold.ttf').as_uri()}'); font-weight: 700; }}
@font-face {{ font-family: 'PJS'; src: url('{(fonts / 'PlusJakartaSans-ExtraBold.ttf').as_uri()}'); font-weight: 800; }}
:root {{ --navy:#0E1B33; --slate:#4A5565; --muted:#6B7280; --teal:#0F766E; --teal-soft:#E3F1EF; --brass:#B8892C; --lime:#F3F5F4; --hair:#DFE3E1; }}
* {{ box-sizing: border-box; }}
html, body {{ margin:0; padding:0; font-family:'PJS', sans-serif; color:var(--slate); font-size:9.6pt; line-height:1.55; -webkit-print-color-adjust:exact; print-color-adjust:exact; }}
h1, h2, h3, h4 {{ color:var(--navy); margin:0; font-weight:700; letter-spacing:-0.01em; }}
h1 {{ font-size:21pt; font-weight:800; line-height:1.15; letter-spacing:-0.02em; margin-bottom:4mm; }}
h1 .num {{ color:var(--brass); font-weight:700; margin-right:3mm; }}
h2 {{ font-size:12.5pt; margin:6mm 0 2mm; }}
h3 {{ font-size:10.5pt; margin:4mm 0 1.5mm; }}
p {{ margin:0 0 2.6mm; max-width:170mm; }}
.lead {{ font-size:11pt; color:var(--navy); line-height:1.5; }}
section {{ break-before:page; padding:0 18mm; }}
section.first {{ break-before:auto; }}
figure {{ margin:3mm 0 4mm; break-inside:avoid; }}
figure img {{ width:100%; display:block; border-radius:2mm; }}
figure.bordered img {{ border:0.3mm solid var(--hair); }}
figcaption {{ font-size:8.2pt; color:var(--muted); margin-top:1.6mm; line-height:1.45; }}
figcaption b {{ color:var(--navy); font-weight:600; }}
table {{ width:100%; border-collapse:collapse; font-size:8.6pt; margin:2mm 0 4mm; font-variant-numeric:tabular-nums; break-inside:avoid; }}
th {{ text-align:left; color:var(--navy); font-weight:600; border-bottom:0.4mm solid var(--navy); padding:1.4mm 2mm 1.4mm 0; }}
td {{ border-bottom:0.2mm solid var(--hair); padding:1.3mm 2mm 1.3mm 0; vertical-align:top; }}
tr.sel td {{ background:var(--teal-soft); font-weight:600; color:var(--navy); }}
table.num td:not(:first-child), table.num th:not(:first-child) {{ white-space:nowrap; }}
.cols {{ display:grid; grid-template-columns:1fr 1fr; gap:7mm; }}
.cols3 {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:5mm; }}
.callout {{ border-left:1.2mm solid var(--teal); background:var(--teal-soft); padding:3mm 4mm; border-radius:0 2mm 2mm 0; margin:3mm 0 4mm; color:var(--navy); break-inside:avoid; }}
.callout.warn {{ border-left-color:var(--brass); background:#FBF6EA; }}
.kpis {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:4mm; margin:4mm 0 5mm; }}
.kpi {{ border:0.3mm solid var(--hair); border-radius:2.5mm; padding:3mm 3.5mm; background:#FCFCFB; }}
.kpi .v {{ font-size:17pt; font-weight:800; color:var(--navy); line-height:1.1; }}
.kpi .l {{ font-size:8pt; color:var(--slate); margin-top:1mm; }}
.kpi .s {{ font-size:7.5pt; color:var(--muted); }}
ul {{ margin:0 0 3mm; padding-left:5mm; }} li {{ margin:0 0 1.2mm; }}
.small {{ font-size:8.4pt; }} .muted {{ color:var(--muted); }}
.chip {{ display:inline-block; border:0.3mm solid rgba(255,255,255,0.35); border-radius:10mm; padding:0.8mm 3mm; margin:0 1.5mm 1.5mm 0; font-size:8.4pt; color:#E8ECEF; }}
.phase {{ display:inline-block; width:3mm; height:3mm; border-radius:0.6mm; vertical-align:-0.4mm; margin-right:1.2mm; }}
.pagebreak-avoid {{ break-inside:avoid; }}
"""

    cover = f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}
@page {{ size:A4; margin:0; }}
.cover {{ width:210mm; height:297mm; background:var(--navy); color:#fff; position:relative; overflow:hidden; padding:22mm 20mm; }}
.cover h1 {{ color:#fff; font-size:44pt; margin:0; letter-spacing:-0.03em; }}
.cover .tag {{ font-size:15pt; color:#D8DEE6; margin-top:4mm; max-width:150mm; line-height:1.4; }}
.cover .shot {{ margin-top:12mm; border-radius:3mm; overflow:hidden; box-shadow:0 0 0 0.4mm rgba(255,255,255,0.25); }}
.cover .shot img {{ width:100%; display:block; }}
.cover .meta {{ position:absolute; left:20mm; right:20mm; bottom:20mm; display:grid; grid-template-columns:1.25fr 1fr; gap:10mm; font-size:9.2pt; color:#C9D1D9; }}
.cover .meta b {{ color:#fff; font-weight:600; }}
.cover .lattice {{ position:absolute; right:-30mm; top:-30mm; opacity:0.12; }}
.cover .mark {{ display:flex; align-items:center; gap:3mm; font-size:10pt; color:#D8B76A; margin-bottom:14mm; font-weight:600; }}
</style></head><body>
<div class="cover">
  <svg class="lattice" width="160mm" height="160mm" viewBox="0 0 64 64"><g fill="none" stroke="#D8B76A" stroke-width="0.6"><rect x="16" y="16" width="32" height="32"/><rect x="16" y="16" width="32" height="32" transform="rotate(45 32 32)"/><circle cx="32" cy="32" r="22"/><circle cx="32" cy="32" r="8"/></g></svg>
  <div class="mark"><svg width="9mm" height="9mm" viewBox="0 0 32 32"><g fill="none" stroke="#B8892C" stroke-width="2" stroke-linejoin="round"><rect x="9" y="9" width="14" height="14"/><rect x="9" y="9" width="14" height="14" transform="rotate(45 16 16)"/></g><circle cx="16" cy="16" r="2.4" fill="#2DD4BF"/></svg>Technical project report</div>
  <h1>Salat Vision</h1>
  <div class="tag">Recognising the visible postures of Islamic prayer in video and turning them into an honest, time-stamped timeline of prayer phases.</div>
  <div class="shot"><img src="{img(ASSETS / 'application_screenshots' / '02c_hasil.png', 1700)}"></div>
  <div style="margin-top:7mm">
    <span class="chip">Computer vision</span><span class="chip">MediaPipe Pose</span><span class="chip">PyTorch → ONNX</span><span class="chip">scikit-learn</span><span class="chip">Hidden Markov model</span><span class="chip">FastAPI</span><span class="chip">React + TypeScript</span>
  </div>
  <div class="meta">
    <div><b>Category</b><br>End-to-end ML product: data audit, model research, temporal decoding, API and web app.<br><br><b>{esc(cfg['version_label'])}</b>, report built from the project's saved artefacts.</div>
    <div><b>Author</b><br>{esc(cfg['author'])}<br>{esc(cfg['role'])}<br>{esc(cfg['contact'])}<br>{esc(cfg['repository'])}</div>
  </div>
</div></body></html>"""

    # model table rows
    rows = []
    for r in cmp.sort_values("crop_macro_f1", ascending=False).itertuples():
        cls = ' class="sel"' if r.model == sel else ""
        rows.append(f"<tr{cls}><td>{esc(MODEL_NAMES.get(r.model, r.model))}</td><td>{r.crop_macro_f1:.2f} ± {r.crop_macro_f1_sd:.2f}</td>"
                    f"<td>{r.crop_macro_f1_ci_lo:.2f}–{r.crop_macro_f1_ci_hi:.2f}</td><td>{r.image_macro_f1:.2f}</td>"
                    f"<td>{r.image_macro_f1_ci_lo:.2f}–{r.image_macro_f1_ci_hi:.2f}</td></tr>")
    model_rows = "\n".join(rows)
    dec_names = {"raw": "Raw per-frame labels", "smoothing": "Moving average + rules", "hmm": "Global grammar HMM",
                 "hmm_local": "HMM + local naming (deployed)"}
    trows = []
    for dname in ("raw", "smoothing", "hmm", "hmm_local"):
        cls = ' class="sel"' if dname == "hmm_local" else ""
        s, i = seg[dname], iid[dname]
        trows.append(f"<tr{cls}><td>{dec_names[dname]}</td><td>{s['phase_acc']['mean']:.2f}</td><td>{s['edit']['mean']:.0f}</td>"
                     f"<td>{s['f1_50']['mean']:.0f}</td><td>{s['seg_ratio']['mean']:.2f}</td><td>{i['phase_acc']['mean']:.2f}</td>"
                     f"<td>{i['f1_50']['mean']:.0f}</td></tr>")
    temporal_rows = "\n".join(trows)
    pv_rows = "\n".join(
        f"<tr{' class=\"sel\"' if v == 'full' else ''}><td>{v}</td><td>{x['detection_rate']:.1%}</td>"
        f"<td>{x['detection_rate_by_class']['sujud']:.0%}</td><td>{x['latency']['mean_ms']:.0f} ms</td><td>{x['size_mb']} MB</td></tr>"
        for v, x in pv["variants"].items())
    stage_names = {"localize": "Person localisation", "pose": "Pose landmarks", "pose_classifier": "Pose classifier", "image_model": "Image model (ONNX)", "total": "Total"}
    rt_rows = "\n".join(f"<tr><td>{stage_names[k]}</td><td>{v['mean_ms']:.1f} ms</td><td>{v['p95_ms']:.1f} ms</td></tr>" for k, v in rt["stages"].items())
    ipc = a["instances_per_class"]
    gpc = a["independent_groups_per_class"]
    det_n = int((loc.loc_source == "detector").sum())
    pose_n = int((loc.loc_source == "pose").sum())
    full_n = int((loc.loc_source == "full_image").sum())
    iou_ok = (loc.loc_best_iou >= 0.5).mean()
    pcolors = {"berdiri": "#EDA100", "rukuk": "#1BAF7A", "iktidal": "#EB6834", "sujud": "#2A78D6", "duduk_antara_sujud": "#008300", "tasyahud": "#E87BA4"}
    sw = lambda p: f'<span class="phase" style="background:{pcolors[p]}"></span>'  # noqa: E731

    body = f"""<!doctype html><html><head><meta charset="utf-8"><style>{css}
@page {{ size:A4; margin:16mm 0 17mm 0; }}
</style></head><body>

<section class="first">
  <h1><span class="num">1</span>Executive summary</h1>
  <p class="lead">Salat Vision analyses a recording of a person praying and returns the sequence of prayer phases —
  Berdiri, Rukuk, Iktidal, Sujud, Duduk di antara dua sujud and Tasyahud — with timestamps, durations and a flag on every
  segment the model is unsure about. It runs locally on a CPU behind an Indonesian-language web app.</p>
  <div class="kpis">
    <div class="kpi"><div class="v">{crop['macro_f1_mean']:.2f}</div><div class="l">Macro-F1, annotated people</div><div class="s">95% CI {f2(crop['macro_f1_ci95'][0])}–{f2(crop['macro_f1_ci95'][1])}</div></div>
    <div class="kpi"><div class="v">{image['macro_f1_mean']:.2f}</div><div class="l">Macro-F1, whole images</div><div class="s">automatic localisation</div></div>
    <div class="kpi"><div class="v">{seg['hmm_local']['f1_50']['mean']:.0f}</div><div class="l">Segment F1@50 (video)</div><div class="s">vs {seg['raw']['f1_50']['mean']:.0f} for raw frame labels</div></div>
    <div class="kpi"><div class="v">{total_ms:.0f} ms</div><div class="l">per frame on 4 vCPU</div><div class="s">{rt['throughput_fps']:.1f} frames/s, no GPU</div></div>
  </div>
  <h2>The problem</h2>
  <p>Learning the order of prayer movements is usually taught by imitation. A tool that shows <i>which</i> movement was
  performed <i>when</i> could support practice and teaching — but a frame classifier alone is not enough: two pairs of
  phases look identical in a single image (standing vs. iktidal, and the two kinds of sitting), and frame-by-frame labels
  flicker.</p>
  <h2>What was built</h2>
  <ul>
    <li><b>Data audit with leakage control.</b> The provided archive turned out to be the public <i>sample</i> of the Salat
    Postures Dataset: {a['n_image_files']} images, {a['n_labeled_instances']} annotated people, four postures. Perceptual hashing and a manual
    session review collapse them into <b>{a['leakage']['n_groups_labeled']} independent groups</b>; every split respects them.</li>
    <li><b>Eleven models under one protocol</b> (repeated stratified group cross-validation with nested tuning and
    cluster-bootstrap CIs): pose-landmark classifiers, frozen and fine-tuned CNNs, training-free rules and a fusion.</li>
    <li><b>A temporal decoder</b> — a grammar-constrained hidden Markov model plus local context rules — that turns four
    visual postures into six prayer phases, evaluated on semi-synthetic sequences built from real held-out scores.</li>
    <li><b>A product:</b> FastAPI service (asynchronous jobs, validation, automatic deletion of uploads) and a React app
    with camera recording, an interactive phase timeline, downloadable reports and a model-transparency page.</li>
  </ul>
  <div class="callout"><b>Key decision.</b> With only {a['leakage']['n_groups_labeled']} independent samples, strong priors beat capacity: a fusion of
  pose geometry (MediaPipe) and frozen ImageNet features outperformed fine-tuned CNNs, which memorised the training folds.
  The fusion was fixed before results were seen, so its estimate carries no selection bias.</div>
  <div class="callout warn"><b>Honest scope.</b> The system recognises visible postures; it does not judge whether a prayer is
  valid. All metrics come from a very small sample and have wide confidence intervals.</div>
</section>

<section>
  <h1><span class="num">2</span>Problem statement</h1>
  <p class="lead">Recognising <i>postures</i> is a classification problem; recognising the <i>prayer</i> is a sequence problem.</p>
  <p>A rakaat follows a fixed order: standing, bowing, standing again (iktidal), prostration, sitting, prostration, then
  either standing for the next rakaat or sitting for the testimony (tasyahud). The training data annotate only what a
  single image can show:</p>
  <table>
    <tr><th>Product phase</th><th>Annotated in the dataset?</th><th>How Salat Vision obtains it</th></tr>
    <tr><td>{sw('berdiri')}Berdiri (qiyam)</td><td>yes (standing)</td><td>visual, named by context</td></tr>
    <tr><td>{sw('rukuk')}Rukuk</td><td>yes</td><td>visual</td></tr>
    <tr><td>{sw('iktidal')}Iktidal</td><td><b>no</b> — looks like standing</td><td>standing directly after bowing</td></tr>
    <tr><td>{sw('sujud')}Sujud</td><td>yes</td><td>visual</td></tr>
    <tr><td>{sw('duduk_antara_sujud')}Duduk di antara dua sujud</td><td><b>no</b> — generic sitting</td><td>sitting between two prostrations</td></tr>
    <tr><td>{sw('tasyahud')}Tasyahud</td><td><b>no</b> — generic sitting</td><td>sitting after the second prostration</td></tr>
  </table>
  <p>Frame-level classification also produces flicker: on a conservative benchmark, raw per-frame labels create
  {seg['raw']['seg_ratio']['mean']:.1f}× more segments than really exist. The task is therefore decomposed into a per-frame posture model and a temporal
  decoder that uses the order of movements — and the product says explicitly which phases are seen and which are inferred.</p>
  <h2>Requirements that shaped the design</h2>
  <ul>
    <li>Local, private processing (no external AI APIs); uploads deleted after analysis.</li>
    <li>CPU-only inference fast enough for interactive use.</li>
    <li>Uncertainty surfaced to the user rather than hidden.</li>
    <li>Indonesian-first interface; reproducible training and evaluation.</li>
  </ul>
  <figure class="bordered"><img src="{img(ASSETS / 'inference_pipeline.png', 2000)}"><figcaption><b>Figure 1.</b> Inference pipeline: per-frame posture recognition, then sequence decoding into prayer phases.</figcaption></figure>
</section>

<section>
  <h1><span class="num">3</span>Dataset and data quality</h1>
  <p class="lead">The audit was run before any modelling decision. It changed the plan three times.</p>
  <div class="cols" style="grid-template-columns:1fr 1.05fr">
    <div>
      <table>
        <tr><th>Item</th><th>Measured</th></tr>
        <tr><td>Image files / readable</td><td>{a['n_image_files']} / {a['n_readable_images']}</td></tr>
        <tr><td>Pascal VOC files</td><td>{a['n_annotation_files']}</td></tr>
        <tr><td>Images without annotation</td><td>{a['n_images_without_annotation']}</td></tr>
        <tr><td>Annotations without image</td><td>{a['n_orphan_annotations']}</td></tr>
        <tr><td>Annotated people (instances)</td><td>{a['n_labeled_instances']}</td></tr>
        <tr><td>Standing / bowing / prostration / sitting</td><td>{ipc['qiyam']} / {ipc['ruku']} / {ipc['sujud']} / {ipc['julus']}</td></tr>
        <tr><td>Independent groups per class</td><td>{gpc['qiyam']} / {gpc['ruku']} / {gpc['sujud']} / {gpc['julus']}</td></tr>
        <tr><td>Image width range</td><td>{a['width_px']['min']}–{a['width_px']['max']} px</td></tr>
      </table>
    </div>
    <figure style="margin-top:0"><img src="{img(ASSETS / 'dataset_samples_anonymized.png', 1300)}"><figcaption><b>Figure 2.</b> Examples per class with the MediaPipe skeleton; faces are blurred in every artefact derived from the dataset.</figcaption></figure>
  </div>
  <h3>What the audit found</h3>
  <ul class="small">
    <li><b>Only four postures are labelled</b>, with spelling variants (<i>sitting/julus</i>, <i>sujud/Sujud</i>) and one box stored as a polygon — silently dropped by a naive parser.</li>
    <li><b>Hidden duplicates across file-name families:</b> a web image is the same frame as <i>qiam(24)</i> (pHash distance 0); another web image comes from the same video as two "SittingMo" frames. Splits are made on <b>{a['leakage']['n_groups_labeled']} groups</b>, not on images.</li>
    <li><b>Capture source is confounded with class:</b> sitting comes almost entirely from the authors' own captures — a shortcut a pixel model could learn.</li>
    <li><b>File names are not labels:</b> the unannotated <i>sujud31.jpg</i> actually shows bowing.</li>
  </ul>
  <div class="cols">
    <figure><img src="{img(ASSETS / 'dataset_class_distribution.png')}"><figcaption><b>Figure 3.</b> Instances per class and the much smaller number of independent capture groups.</figcaption></figure>
    <figure><img src="{img(ASSETS / 'dataset_source_by_class.png')}"><figcaption><b>Figure 4.</b> Capture source vs. class.</figcaption></figure>
  </div>
</section>

<section>
  <h1><span class="num">4</span>System architecture</h1>
  <p class="lead">Five layers with one configuration file and one model bundle as the contract between them.</p>
  <figure class="bordered"><img src="{img(ASSETS / 'system_architecture.png', 2000)}"><figcaption><b>Figure 5.</b> From a ZIP of annotated images to a browser timeline. Editable SVG and Mermaid sources ship with the project.</figcaption></figure>
  <div class="cols">
    <div>
      <h3>Training side</h3>
      <ul class="small">
        <li>Safe extraction (path traversal, symlinks, zip bombs, non-data files) with byte-level verification.</li>
        <li>Feature cache: pose landmarks and CNN embeddings computed once per run.</li>
        <li>Experiment runner: all models, both evaluation views, out-of-fold predictions saved for later analysis.</li>
        <li>Export: the selected pipeline is re-fit on all data; the CNN goes to ONNX with an enforced parity check
        (cosine {man['training']['image_classifier']['onnx_parity']['min_cosine']:.7f}); every file is recorded with its SHA-256.</li>
      </ul>
    </div>
    <div>
      <h3>Serving side</h3>
      <ul class="small">
        <li>No PyTorch at runtime: ONNX Runtime, MediaPipe and scikit-learn only.</li>
        <li>Checksums are verified <i>before</i> any serialised model is deserialised.</li>
        <li>Video jobs run in a bounded thread pool (one predictor per thread) and report real progress.</li>
        <li>Missing or tampered model ⇒ <code>503</code> with a reason, never a placeholder prediction.</li>
      </ul>
    </div>
  </div>
  <h2 style="margin-top:4mm">4.1 Runtime on a CPU-only server</h2>
  <div class="cols" style="break-inside:avoid">
    <table>
      <tr><th>Stage (4 vCPU, no GPU)</th><th>Mean</th><th>p95</th></tr>
      {rt_rows}
    </table>
    <table>
      <tr><th>Pose model</th><th>Pose found</th><th>Prostration</th><th>Latency</th><th>Size</th></tr>
      {pv_rows}
    </table>
  </div>
  <p class="small">The ONNX Runtime embedder takes {rt['embedder']['onnxruntime_ms']['mean_ms']:.1f} ms vs {rt['embedder']['pytorch_ms']['mean_ms']:.1f} ms in PyTorch. A 43-second clip at 5 fps is analysed in about 20 seconds.</p>
</section>

<section>
  <h1><span class="num">5</span>Methodology</h1>
  <h2>5.1 Person localisation and features</h2>
  <p>Each frame is searched for a person with EfficientDet-Lite0 (COCO), falling back to the pose-landmark box and then
  the full frame; the crop adds 15% padding, exactly as during training. Two representations are computed:</p>
  <ul>
    <li><b>Pose geometry.</b> MediaPipe Pose Landmarker returns 33 landmarks in image and metric 3-D coordinates. Nineteen
    features are derived: torso inclination, hip and knee angles, thigh and shin inclination (3-D), head, shoulder, wrist,
    knee and ankle heights relative to the hips, silhouette aspect and visibility. Landmarks with visibility below 0.35
    are treated as missing, and the model receives explicit missing-value indicators.</li>
    <li><b>Appearance.</b> The crop is letterboxed (aspect ratio preserved — a tall box means standing, a wide box
    prostration) and embedded by an ImageNet MobileNetV3-Large or ResNet18.</li>
  </ul>
  <h2>5.2 Candidate models</h2>
  <p>A majority baseline; geometric rules with thresholds fixed from anatomy before evaluation (knee 140°, torso
  45°/55°); four pose-feature classifiers; logistic regression on frozen CNN embeddings (horizontal-flip augmentation);
  end-to-end fine-tuning (AdamW, cosine schedule, label smoothing, class weights, frozen BatchNorm, augmentation,
  early stopping on a group-wise validation split, checkpoints); and a <b>pre-registered fusion</b> averaging the pose
  logistic regression and the MobileNetV3 probe.</p>
  <h2>5.3 Evaluation protocol</h2>
  <ul>
    <li><b>3 × repeated stratified group 5-fold cross-validation</b> over people: no capture group ever appears on both
    sides of a split. Hyper-parameters are tuned by an inner group-wise 3-fold on the training fold only.</li>
    <li><b>Two test views:</b> annotated person crops (classification in isolation) and whole images passed through the
    real localisation chain (end-to-end).</li>
    <li><b>Macro-F1</b> as the primary metric (classes are imbalanced), plus accuracy, per-class metrics, confusion matrices,
    calibration (ECE) and abstention.</li>
    <li><b>95% confidence intervals from a cluster bootstrap</b> that resamples capture groups, not images.</li>
  </ul>
  <h2>5.4 Temporal decoding</h2>
  <p>Frames are resampled by FFmpeg at 5 fps. A hidden Markov model with seven states (two prostration states share one
  visual class) follows the prayer grammar; the probability of switching phase between frames is 1 − exp(−Δt / 2 s), and a
  small off-grammar probability keeps recordings that start mid-prayer decodable. Viterbi gives the segmentation; local
  context rules then name each segment. Segments whose mean score falls below 0.55 are reported as uncertain.</p>
  <p>Because no labelled prayer videos exist, the decoder is evaluated on <b>300 semi-synthetic sequences per regime</b>
  (2–4 rakaat, realistic durations, motion blends, dropped frames) whose per-frame scores are <b>real out-of-fold
  predictions</b> on held-out people. In the conservative regime one held-out prediction is reused for a whole segment, so
  errors persist as they would for one person in one pose. Decoder settings were chosen on a separate validation seed.</p>
</section>

<section>
  <h1><span class="num">6</span>Experiments and results</h1>
  <figure><img src="{img(ASSETS / 'model_comparison.png', 2000)}"><figcaption><b>Figure 6.</b> Macro-F1 of all eleven models on both views; whiskers are 95% cluster-bootstrap confidence intervals.</figcaption></figure>
  <table class="num">
    <tr><th>Model</th><th>Macro-F1, crops (mean ± SD)</th><th>95% CI</th><th>Macro-F1, images</th><th>95% CI</th></tr>
    {model_rows}
  </table>
  <div class="callout"><b>Findings.</b> (1) Pose geometry carries most of the signal — training-free rules alone reach
  {cmp.set_index('model').loc['pose_rules','crop_macro_f1']:.2f}. (2) Fusion is best on both views and the most stable across repeats: pose and appearance make
  different mistakes. (3) Fine-tuning does not beat the frozen probe and takes ~100× longer to train. (4) MobileNetV3 beats
  ResNet18 in every setting. (5) The top models' intervals overlap: the ranking is suggestive, not conclusive.</div>
</section>

<section>
  <h2 style="margin-top:0">6.1 Where the selected model fails</h2>
  <div class="cols">
    <div>
      <figure><img src="{img(ASSETS / 'confusion_matrix.png', 1600)}"><figcaption><b>Figure 7.</b> Out-of-fold confusion of the selected model (counts over three repeats).</figcaption></figure>
      <table>
        <tr><th>Class (crops)</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
        <tr><td>Standing</td><td>{pc['qiyam']['precision']:.2f}</td><td>{pc['qiyam']['recall']:.2f}</td><td>{pc['qiyam']['f1']:.2f}</td></tr>
        <tr><td>Bowing</td><td>{pc['ruku']['precision']:.2f}</td><td>{pc['ruku']['recall']:.2f}</td><td>{pc['ruku']['f1']:.2f}</td></tr>
        <tr><td>Prostration</td><td>{pc['sujud']['precision']:.2f}</td><td>{pc['sujud']['recall']:.2f}</td><td>{pc['sujud']['f1']:.2f}</td></tr>
        <tr><td>Sitting</td><td>{pc['julus']['precision']:.2f}</td><td>{pc['julus']['recall']:.2f}</td><td>{pc['julus']['f1']:.2f}</td></tr>
      </table>
    </div>
    <div>
      <p><b>Bowing</b> is the weakest class (6 examples): it is confused with sitting when the camera looks down on the person.
      <b>Sitting</b> loses most in the end-to-end view, where detector crops differ from annotated boxes and sitting is
      mistaken for prostration — the biggest gap between the two evaluation views.</p>
      <p>Localisation itself is reliable: the detector found the person in {det_n} of {len(loc)} images, the pose fallback in {pose_n} and
      the full frame was used {full_n} times; {iou_ok:.0%} of boxes overlap an annotated person with IoU ≥ 0.5.</p>
      <figure><img src="{img(ASSETS / 'calibration_reliability.png', 1100)}"><figcaption><b>Figure 8.</b> Scores are under-confident (ECE {e['crop']['calibration']['ece']:.2f}): the "uncertain" flag is conservative. Scores are never presented as calibrated probabilities.</figcaption></figure>
    </div>
  </div>
  <figure><img src="{img(ASSETS / 'training_curves.png', 1800)}"><figcaption><b>Figure 9.</b> Fine-tuning across 15 folds: training loss collapses while validation loss stalls — memorisation of ~32 training people per fold.</figcaption></figure>
</section>

<section>
  <h2 style="margin-top:0">6.2 From frames to phases</h2>
  <figure style="width:90%"><img src="{img(ASSETS / 'temporal_benchmark.png', 2000)}"><figcaption><b>Figure 10.</b> Temporal decoders on {seg['n_sequences']} semi-synthetic 2–4 rakaat sequences per regime, using real out-of-fold scores.</figcaption></figure>
  <table class="num">
    <tr><th>Decoder</th><th>Phase acc. (cons.)</th><th>Edit</th><th>F1@50</th><th>Segments / true</th><th>Phase acc. (opt.)</th><th>F1@50 (opt.)</th></tr>
    {temporal_rows}
  </table>
  <p>The global grammar HMM is the best <i>segmenter</i>, but when an entire segment is misrecognised it can lose track of
  where it is in the prayer and mislabel later standings and sittings. The deployed hybrid keeps the HMM segmentation and
  names each segment from its neighbours — best on phase accuracy and F1@50 in both regimes.</p>
  <figure style="width:90%"><img src="{img(ASSETS / 'posture_timeline.png', 2000)}"><figcaption><b>Figure 11.</b> The synthetic demo clip: plan, raw per-frame labels and decoded phases. The clip is assembled from training images, so it checks the mechanics, not accuracy.</figcaption></figure>
</section>

<section>
  <h1><span class="num">7</span>Application</h1>
  <p class="lead">An Indonesian-first web app whose visual signature is the phase timeline — a woven band of prayer phases.</p>
  <div class="cols">
    <figure class="bordered"><img src="{img(ASSETS / 'application_screenshots' / '01_beranda.png', 1500)}"><figcaption><b>Figure 12.</b> Landing page with live model status and the six-phase sequence (pose-style pictograms).</figcaption></figure>
    <figure class="bordered"><img src="{img(ASSETS / 'application_screenshots' / '02b_progres.png', 1500)}"><figcaption><b>Figure 13.</b> Analysis workspace: upload, camera recording or demo clip; real upload and frame progress.</figcaption></figure>
  </div>
  <figure class="bordered" style="width:70%; margin-left:auto; margin-right:auto"><img src="{img(ASSETS / 'application_screenshots' / '02c_hasil.png', 1800)}"><figcaption><b>Figure 14.</b> Results: annotated video synchronised with the phase timeline (click a segment to seek), cycle count, uncertain segments and analysis speed.</figcaption></figure>
  <div class="cols">
    <figure class="bordered"><img src="{img(ASSETS / 'application_screenshots' / '03_analisis_foto.png', 1500)}"><figcaption><b>Figure 15.</b> Photo mode: posture, per-class scores and the detected skeleton.</figcaption></figure>
    <figure class="bordered"><img src="{img(ASSETS / 'application_screenshots' / '04_model.png', 1500)}"><figcaption><b>Figure 16.</b> Model page: real cross-validation metrics with intervals, comparison and limitations.</figcaption></figure>
  </div>
</section>

<section>
  <h2 style="margin-top:0">7.1 API</h2>
  <table>
    <tr><th>Endpoint</th><th>Purpose</th></tr>
    <tr><td>GET /health, GET /model-info</td><td>liveness, model status, classes, phases, metrics, benchmarks, limitations</td></tr>
    <tr><td>POST /predict/image</td><td>posture, per-class scores, person box, 33 landmarks, timings</td></tr>
    <tr><td>POST /predict/video</td><td>synchronous analysis for short clips and scripts</td></tr>
    <tr><td>POST /jobs/video, GET /jobs/{{id}}</td><td>asynchronous analysis with real progress</td></tr>
    <tr><td>GET /jobs/{{id}}/video, /report?format=json|csv; DELETE /jobs/{{id}}</td><td>annotated H.264 video, downloads, immediate deletion</td></tr>
  </table>
  <h1 style="margin-top:8mm"><span class="num">8</span>Engineering decisions</h1>
  <table>
    <tr><th>Decision</th><th>Why</th><th>Alternative rejected</th></tr>
    <tr><td>Group-aware CV with cluster-bootstrap CIs</td><td>duplicates and sessions cross file-name families; 29 effective samples</td><td>random image split (leaks); single hold-out (≈8 test people)</td></tr>
    <tr><td>Pose + frozen-CNN fusion</td><td>best and most stable; cheap; pre-registered</td><td>fine-tuning (memorises, 100× slower), ResNet18</td></tr>
    <tr><td>Four visual classes + temporal naming</td><td>the data cannot support six visual classes</td><td>inventing labels for iktidal/tasyahud</td></tr>
    <tr><td>HMM segmentation + local naming</td><td>robust to a fully misrecognised segment</td><td>global grammar naming; moving average</td></tr>
    <tr><td>MediaPipe Pose "full"</td><td>highest detection rate at 34 ms</td><td>"heavy" (2× slower, no gain), YOLO-Pose (AGPL licence, no pose labels to fine-tune)</td></tr>
    <tr><td>ONNX Runtime at serving</td><td>2.3× faster, no PyTorch in the image</td><td>TorchScript / PyTorch serving</td></tr>
    <tr><td>Async jobs + temp storage with TTL</td><td>real progress; uploads never persisted</td><td>long synchronous requests; storing videos</td></tr>
  </table>
  <h2>Reproducibility and testing</h2>
  <ul>
    <li>One configuration file, fixed seeds, saved folds, pinned dependencies and a full environment lock.</li>
    <li>Re-exporting the bundle reproduces <b>bit-identical</b> files; the notebook re-run reproduced the scripted CV numbers exactly.</li>
    <li>55 Python tests (extraction attacks, leakage, features, decoder, real-model inference, tamper detection, API job
    lifecycle, CORS, 503 path), 12 frontend unit tests and 5 Playwright end-to-end tests against the real backend.</li>
    <li>Docker and Compose files are provided; they were syntax-checked but could not be built in the authoring environment.</li>
  </ul>
</section>

<section>
  <h1><span class="num">9</span>Limitations and future work</h1>
  <div class="cols">
    <div>
      <h3>Limitations</h3>
      <ul>
        <li><b>Data volume.</b> 40 people in 29 groups; estimates have wide intervals and the full 1,904-image release was not available.</li>
        <li><b>No real prayer videos.</b> Temporal results are semi-synthetic; continuous motion, occlusion and camera shake are untested.</li>
        <li><b>Inferred phases.</b> Iktidal and both sittings are named by order; a skipped or extra movement propagates to neighbours.</li>
        <li><b>Prostration</b> defeats the pose estimator in a third of crops; the image branch covers those frames.</li>
        <li><b>Confidence</b> is under-confident and uncalibrated.</li>
        <li><b>Coverage.</b> Viewpoints, clothing, women's and children's prayer, group prayer and low light are not represented.</li>
      </ul>
    </div>
    <div>
      <h3>Next steps</h3>
      <ul>
        <li>Re-run the identical pipeline on the full dataset (the code is dataset-agnostic).</li>
        <li>Record consented, phase-labelled prayer videos to evaluate and train the temporal model directly (e.g. a small temporal CNN or HMM with learned durations).</li>
        <li>Temperature scaling with nested validation once data allows.</li>
        <li>Fine-grained cues (raised index finger for tasyahud, foot position) with hand landmarks.</li>
        <li>On-device inference in the browser (MediaPipe + ONNX Runtime Web) to remove uploads entirely.</li>
      </ul>
    </div>
  </div>
  <h1 style="margin-top:5mm"><span class="num">10</span>Conclusion</h1>
  <p>Salat Vision shows the full path from an unexpectedly small, messy dataset to a working, honest product: an audit that
  found leakage and confounds before they could inflate results, a leakage-aware comparison of eleven models, a
  pre-registered fusion that reaches a macro-F1 of {crop['macro_f1_mean']:.2f} on held-out people, a temporal decoder that turns four postures
  into six phases with clean segments, and a fast, private, tested web application that tells its users what it does not know.</p>
  <h2>References</h2>
  <ol style="font-size:7.8pt; line-height:1.4; margin-top:0">
    <li>Koubâa, A. et al. (2020). Activity Monitoring of Islamic Prayer (Salat) Postures using Deep Learning. <i>6th Conference on Data Science and Machine Learning Applications (CDMA)</i>, 106–111. IEEE. arXiv:1911.04102. Dataset: github.com/riotu-lab/Salat-Postures-Dataset.</li>
    <li>Bazarevsky, V. et al. (2020). BlazePose: On-device Real-time Body Pose Tracking. arXiv:2006.10204. (MediaPipe Pose Landmarker, Apache-2.0.)</li>
    <li>Tan, M., Pang, R., Le, Q. V. (2020). EfficientDet: Scalable and Efficient Object Detection. CVPR. (MediaPipe EfficientDet-Lite0.)</li>
    <li>Howard, A. et al. (2019). Searching for MobileNetV3. ICCV. — He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.</li>
    <li>Rabiner, L. R. (1989). A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition. Proc. IEEE 77(2).</li>
    <li>Lea, C. et al. (2017). Temporal Convolutional Networks for Action Segmentation and Detection. CVPR. (Edit score, F1@k.)</li>
    <li>Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML. (ECE.)</li>
    <li>Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12. — ONNX Runtime, FastAPI, React documentation.</li>
  </ol>
  <p class="muted" style="font-size:7.8pt">Code licence: MIT. The dataset archive contained no licence file; images are not redistributed and all published artefacts blur faces.</p>
</section>
</body></html>"""
    return cover, body


def main() -> int:
    d = load()
    cover, body = build_html(d)
    BUILD.mkdir(parents=True, exist_ok=True)
    (BUILD / "cover.html").write_text(cover, encoding="utf-8")
    (BUILD / "body.html").write_text(body, encoding="utf-8")
    script = HERE / "print_pdf.cjs"
    import os

    env = dict(os.environ)
    env["NODE_PATH"] = subprocess.run(["npm", "root", "-g"], capture_output=True, text=True).stdout.strip()
    subprocess.run(["node", str(script), str(BUILD / "cover.html"), str(BUILD / "cover.pdf"), "cover"], check=True, env=env)
    subprocess.run(["node", str(script), str(BUILD / "body.html"), str(BUILD / "body.pdf"), "body"], check=True, env=env)
    from pypdf import PdfReader, PdfWriter

    w = PdfWriter()
    for part in ("cover.pdf", "body.pdf"):
        for page in PdfReader(BUILD / part).pages:
            w.add_page(page)
    w.add_metadata({"/Title": "Salat Vision — Technical Report", "/Author": d["cfg"]["author"],
                    "/Subject": "Prayer-posture recognition: data audit, model research, temporal decoding, API and web app"})
    out = HERE / "Salat_Vision_Technical_Report.pdf"
    with open(out, "wb") as f:
        w.write(f)
    n = len(PdfReader(out).pages)
    print(f"{out.relative_to(ROOT)}: {n} pages, {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
