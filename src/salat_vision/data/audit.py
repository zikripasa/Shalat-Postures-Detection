"""Dataset audit: statistics, quality checks, leakage groups and figures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from salat_vision.config import Config, get_config
from salat_vision.data.dataset import DatasetIndex, build_index
from salat_vision.data.splits import assign_groups, load_session_groups
from salat_vision.labels import VISUAL_CLASSES, VISUAL_LABELS_EN
from salat_vision.viz import INK, INK_2, INK_MUTED, SERIES, SEQ_BLUE, apply_style, hairline_grid, rounded_bars, save

TARGET_PHASES = ["Standing (qiyam)", "Bowing (rukuk)", "Standing after bowing (iktidal)", "Prostration (sujud)",
                 "Sitting between prostrations", "Testimony sitting (tasyahud)"]


def run_audit(cfg: Config | None = None) -> tuple[DatasetIndex, dict]:
    cfg = cfg or get_config()
    idx = build_index(cfg=cfg)
    groups, links = assign_groups(idx.images, cfg)
    idx.images["group_id"] = idx.images["image_id"].map(groups)
    idx.instances["group_id"] = idx.instances["image_id"].map(groups)
    li = idx.labeled_instances
    im = idx.images

    per_class_inst = li["label"].value_counts().reindex(VISUAL_CLASSES, fill_value=0)
    per_class_img = li.groupby("label")["image_id"].nunique().reindex(VISUAL_CLASSES, fill_value=0)
    per_class_groups = li.groupby("label")["group_id"].nunique().reindex(VISUAL_CLASSES, fill_value=0)
    raw_labels = idx.instances["raw_label"].value_counts().to_dict()
    readable = im[im["readable"]]
    group_sizes = li.groupby("group_id").size()

    summary = {
        "dataset_dir": str(idx.dataset_dir.relative_to(cfg.root)) if idx.dataset_dir.is_relative_to(cfg.root) else str(idx.dataset_dir),
        "kind": "image dataset with Pascal VOC bounding boxes (object-detection format); no videos, no sequences",
        "n_image_files": int(len(im)),
        "n_readable_images": int(im["readable"].sum()),
        "n_unreadable_images": int((~im["readable"]).sum()),
        "n_annotation_files": int(im["has_annotation"].sum() + len(idx.orphan_annotations)),
        "n_images_with_annotation": int(im["has_annotation"].sum()),
        "n_images_without_annotation": int((~im["has_annotation"]).sum()),
        "images_without_annotation": im.loc[~im["has_annotation"], "image_id"].tolist(),
        "n_orphan_annotations": int(len(idx.orphan_annotations)),
        "orphan_annotations": idx.orphan_annotations.to_dict(orient="records"),
        "n_labeled_images": int(li["image_id"].nunique()),
        "n_labeled_instances": int(len(li)),
        "n_invalid_instances": int((~idx.instances["valid"]).sum()),
        "raw_label_spellings": raw_labels,
        "classes": VISUAL_CLASSES,
        "instances_per_class": per_class_inst.to_dict(),
        "images_per_class": per_class_img.to_dict(),
        "independent_groups_per_class": per_class_groups.to_dict(),
        "imbalance_ratio_max_min": float(per_class_inst.max() / max(per_class_inst.min(), 1)),
        "multi_person_images": im.loc[im["n_objects"] > 1, ["image_id", "n_objects", "labels"]].to_dict(orient="records"),
        "mixed_label_images": int(im["mixed_labels"].sum()),
        "polygon_annotations": int((li["box_format"] == "polygon").sum()),
        "file_types": im["ext"].value_counts().to_dict(),
        "exif_rotated_images": int((readable["exif_orientation"].fillna(1) != 1).sum()),
        "xml_size_mismatches": int((im["xml_size_matches_stored"] == False).sum()),  # noqa: E712
        "exact_duplicate_files": int(im.duplicated("sha256", keep=False).sum()),
        "width_px": {"min": int(readable["width"].min()), "median": float(readable["width"].median()), "max": int(readable["width"].max())},
        "height_px": {"min": int(readable["height"].min()), "median": float(readable["height"].median()), "max": int(readable["height"].max())},
        "box_rel_area": {"min": float(li["rel_area"].min()), "median": float(li["rel_area"].median()), "max": float(li["rel_area"].max())},
        "box_aspect_hw_median_by_class": li.groupby("label")["aspect_hw"].median().reindex(VISUAL_CLASSES).round(2).to_dict(),
        "source_kinds": im["source_kind"].value_counts().to_dict(),
        "source_by_class": pd.crosstab(li["image_id"].map(im.set_index("image_id")["source_kind"]), li["label"])
                             .reindex(columns=VISUAL_CLASSES, fill_value=0).to_dict(orient="index"),
        "leakage": {
            "n_groups_labeled": int(li["group_id"].nunique()),
            "n_multi_instance_groups": int((group_sizes > 1).sum()),
            "largest_group_instances": int(group_sizes.max()),
            "group_links": links.to_dict(orient="records"),
            "session_groups_reviewed": {k: v.get("members", []) for k, v in load_session_groups(cfg).items()},
        },
        "target_phases_requested": TARGET_PHASES,
        "phase_support": {
            "Standing (qiyam)": "annotated (qiyam)",
            "Bowing (rukuk)": "annotated (ruku)",
            "Prostration (sujud)": "annotated (sujud / Sujud)",
            "Standing after bowing (iktidal)": "NOT annotated — visually identical to standing; resolved from temporal context only",
            "Sitting between prostrations": "NOT annotated separately — annotated as generic sitting (sitting / julus); resolved from temporal context only",
            "Testimony sitting (tasyahud)": "NOT annotated separately — annotated as generic sitting (sitting / julus); resolved from temporal context only",
        },
        "issues": idx.issues,
        "provenance": {
            "source": "riotu-lab/Salat-Postures-Dataset (GitHub) — 'Sample-images-with-annotations' folder only",
            "full_dataset": "1,904 images / 2,156 instances per the dataset README, hosted on Kaggle (riotulab/salat-postures); not available in this environment",
            "citation": "Koubaa, A. et al. (2020). Activity Monitoring of Islamic Prayer (Salat) Postures using Deep Learning. 6th Conference on Data Science and Machine Learning Applications (CDMA), pp. 106-111. IEEE. arXiv:1911.04102",
            "license": "No license file is included in the archive; reuse terms could not be verified. Images are used here for research/education only and are not redistributed.",
        },
    }
    return idx, summary


# ---------------------------------------------------------------------------------- figures
def fig_class_distribution(li: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    counts = li["label"].value_counts().reindex(VISUAL_CLASSES, fill_value=0)
    groups = li.groupby("label")["group_id"].nunique().reindex(VISUAL_CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.set_xlim(0, counts.max() * 1.25)
    ax.set_ylim(-0.6, len(counts) - 0.4)
    ax.invert_yaxis()
    ys = np.arange(len(counts))
    rounded_bars(ax, ys, counts.values.astype(float), color=SERIES[0], width=0.55, horizontal=True)
    ax.set_yticks(ys, [VISUAL_LABELS_EN[c] for c in counts.index])
    for y, (c, g) in zip(ys, zip(counts.values, groups.values)):
        ax.text(c + counts.max() * 0.02, y, f"{c}  ({g} independent groups)", va="center", ha="left", color=INK_2, fontsize=9)
    ax.set_xlabel("Annotated person instances")
    hairline_grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Class distribution — {len(li)} instances in {li['image_id'].nunique()} images")
    return save(fig, out, also_svg=True)


def fig_source_by_class(li: pd.DataFrame, images: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    src = li["image_id"].map(images.set_index("image_id")["source_kind"])
    tab = pd.crosstab(src, li["label"]).reindex(columns=VISUAL_CLASSES, fill_value=0)
    tab = tab.loc[tab.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(tab) + 1.4))
    ax.imshow(tab.values, cmap=SEQ_BLUE, aspect="auto", vmin=0, vmax=max(tab.values.max(), 1))
    ax.set_xticks(range(len(VISUAL_CLASSES)), [VISUAL_LABELS_EN[c].split(" (")[0] for c in VISUAL_CLASSES])
    ax.set_yticks(range(len(tab)), tab.index)
    vmax = tab.values.max()
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="#ffffff" if v > 0.55 * vmax else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Capture source vs. class (instances)")
    ax.text(0, -0.12, "Web images dominate standing/prostration; sitting comes mostly from own captures — a shortcut risk.",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED, va="top")
    return save(fig, out, also_svg=True)


def fig_box_geometry(li: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    rng = np.random.default_rng(0)
    for k, c in enumerate(VISUAL_CLASSES):
        v = np.log2(li.loc[li["label"] == c, "aspect_hw"].values)
        y = k + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(v, y, s=36, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        if len(v):
            ax.plot([np.median(v)] * 2, [k - 0.28, k + 0.28], color=INK, linewidth=2, zorder=4)
    ax.set_yticks(range(len(VISUAL_CLASSES)), [VISUAL_LABELS_EN[c] for c in VISUAL_CLASSES])
    ax.invert_yaxis()
    ticks = [-2, -1, 0, 1, 2]
    ax.set_xticks(ticks, ["1:4", "1:2", "1:1", "2:1", "4:1"])
    ax.axvline(0, color="#c3c2b7", linewidth=0.8, zorder=1)
    ax.set_xlabel("Bounding-box height : width (log scale) — black tick = median")
    hairline_grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_title("Box shape separates postures — but not sitting from bowing")
    return save(fig, out, also_svg=True)


def fig_resolution(images: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    apply_style()
    r = images[images["readable"]]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    lab = r["has_annotation"]
    ax.scatter(r.loc[lab, "width"], r.loc[lab, "height"], s=40, color=SERIES[0], edgecolor="#fcfcfb", linewidth=1.5,
               label="annotated", zorder=3)
    ax.scatter(r.loc[~lab, "width"], r.loc[~lab, "height"], s=40, color=SERIES[1], edgecolor="#fcfcfb", linewidth=1.5,
               label="no annotation", zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Width (px, log)")
    ax.set_ylabel("Height (px, log)")
    hairline_grid(ax, "both")
    ax.legend(loc="upper left")
    ax.set_title("Image resolution varies 20x")
    return save(fig, out, also_svg=True)


def write_markdown(summary: dict, idx: DatasetIndex, out: Path) -> Path:
    s = summary
    lines = [
        "# Dataset audit — Salat Postures (provided sample)",
        "",
        "_Generated by `scripts/audit_dataset.py` from the files actually present in "
        f"`{s['dataset_dir']}`._",
        "",
        "## What the archive contains",
        "",
        f"- **Type:** {s['kind']}.",
        f"- **Image files:** {s['n_image_files']} ({', '.join(f'{k}: {v}' for k, v in s['file_types'].items())}); "
        f"readable: {s['n_readable_images']}, unreadable: {s['n_unreadable_images']}.",
        f"- **Annotation files:** {s['n_annotation_files']} Pascal VOC XML; {s['n_images_with_annotation']} images have one, "
        f"{s['n_images_without_annotation']} do not ({', '.join(s['images_without_annotation'])}), and "
        f"{s['n_orphan_annotations']} XML files have no image.",
        f"- **Labeled data used for modelling:** {s['n_labeled_instances']} person instances in {s['n_labeled_images']} images "
        f"(one annotation uses a 4-point `<polygon>` instead of `<bndbox>`; it is parsed as its bounding rectangle).",
        f"- **Provenance:** {s['provenance']['source']}. The full dataset ({s['provenance']['full_dataset']}).",
        f"- **License:** {s['provenance']['license']}",
        "",
        "## Classes",
        "",
        "| Class | Raw spellings | Instances | Images | Independent groups |",
        "|---|---|---:|---:|---:|",
    ]
    spellings = {c: [] for c in VISUAL_CLASSES}
    from salat_vision.labels import normalize_label
    for raw in s["raw_label_spellings"]:
        lab = normalize_label(raw)
        if lab:
            spellings[lab].append(f"`{raw}`")
    for c in VISUAL_CLASSES:
        lines.append(f"| {VISUAL_LABELS_EN[c]} | {', '.join(spellings[c])} | {s['instances_per_class'][c]} | "
                     f"{s['images_per_class'][c]} | {s['independent_groups_per_class'][c]} |")
    lines += [
        "",
        f"Max/min class ratio: {s['imbalance_ratio_max_min']:.1f}. Multi-person images: "
        f"{len(s['multi_person_images'])}; images mixing classes: {s['mixed_label_images']}.",
        "",
        "### The six target phases",
        "",
        "| Target phase | Support in this dataset |",
        "|---|---|",
    ]
    for k, v in s["phase_support"].items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "Only **four visual postures** are annotated. Iktidal looks like standing and both sittings look alike in a "
        "single frame, so the product never claims to *see* these differences: it infers them from the order of postures "
        "in a video (grammar-constrained temporal decoding).",
        "",
        "## Quality checks",
        "",
        f"- Unreadable/corrupted images: {s['n_unreadable_images']}.",
        f"- EXIF-rotated images: {s['exif_rotated_images']}; XML `<size>` vs. pixel-size mismatches: {s['xml_size_mismatches']}.",
        f"- Exact duplicate files (SHA-256): {s['exact_duplicate_files']}.",
        f"- Invalid boxes (unknown label, out of bounds, < min size): {s['n_invalid_instances']}.",
        f"- Image size: width {s['width_px']['min']}–{s['width_px']['max']} px (median {s['width_px']['median']:.0f}); "
        f"height {s['height_px']['min']}–{s['height_px']['max']} px.",
        f"- Box area relative to image: {s['box_rel_area']['min']:.3f}–{s['box_rel_area']['max']:.3f} "
        f"(median {s['box_rel_area']['median']:.3f}); several people are tiny (< 5% of the frame).",
        "- One 'ruku' image is a cartoon infographic (not a photograph); one 'sujud' image is a footballer on a pitch.",
        "- Unannotated file `sujud31.jpg` actually shows **bowing**: file names are not reliable labels and are never used as labels.",
        "",
        "## Leakage risks and grouping",
        "",
        f"- Near-duplicates (perceptual hash) and same-session/same-video images were linked into **{s['leakage']['n_groups_labeled']} "
        f"independent groups** for the {s['n_labeled_instances']} labeled instances (largest group: "
        f"{s['leakage']['largest_group_instances']} instances).",
        "- Cross-source duplicates exist: a web (Labelbox) image is a frame of the same video as two 'SittingMo' frames, "
        "and another web image is byte-for-byte the same frame as `qiam(24).jpg` (pHash distance 0).",
        "- Folder names in the XML `<path>` field ('vids to images', 'videoplayback') show that part of the data are video frames.",
        "- Every split in this project is a **stratified group k-fold**: a group never appears in both train and test.",
        "",
        "| Link | Reason |",
        "|---|---|",
    ]
    for link in s["leakage"]["group_links"]:
        lines.append(f"| `{link['image_a']}` – `{link['image_b']}` | {link['reason']} |")
    lines += [
        "",
        "## Shortcut-learning risk",
        "",
        "Capture source is confounded with class: sitting images come almost entirely from the authors' own captures "
        "(lab room, two videos), while standing and prostration come mostly from web images. A pixel model can score well by "
        "recognising the *source* instead of the *posture*. Group-aware evaluation and pose-based (background-free) features "
        "are the mitigations used here.",
        "",
        "## Recommendations",
        "",
        f"1. Treat every metric from this sample as a **high-variance estimate** ({s['leakage']['n_groups_labeled']} independent groups). Report confidence intervals.",
        "2. Prefer methods with strong priors (pretrained pose estimation, frozen ImageNet embeddings) over end-to-end fine-tuning.",
        "3. Model **four visual postures**; derive the six prayer phases from temporal order, and label that inference as such.",
        "4. Obtain the full dataset (Kaggle) and, ideally, real prayer videos with phase-level timestamps before any deployment claim.",
        "",
        "## Issues log",
        "",
        "| File | Issue | Detail |",
        "|---|---|---|",
    ]
    for issue in s["issues"]:
        lines.append(f"| `{issue['file']}` | {issue['issue']} | {issue['detail']} |")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def save_tables(idx: DatasetIndex, summary: dict, cfg: Config) -> dict[str, Path]:
    art = cfg.path("artifacts_dir")
    paths = {
        "summary": art / "metrics" / "dataset_audit.json",
        "images": art / "metrics" / "dataset_images.csv",
        "instances": art / "metrics" / "dataset_instances.csv",
    }
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    keep_img = ["image_id", "file", "ext", "bytes", "width", "height", "exif_orientation", "has_annotation", "n_objects",
                "labels", "source_kind", "xml_folder", "group_id", "phash", "sha256", "mean_brightness"]
    idx.images[keep_img].to_csv(paths["images"], index=False)
    idx.instances.to_csv(paths["instances"], index=False)
    return paths
