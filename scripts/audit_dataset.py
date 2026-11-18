#!/usr/bin/env python
"""Audit the dataset and write tables, figures, a Markdown report and the CV splits.

Outputs
  artifacts/metrics/dataset_audit.json      summary statistics
  artifacts/metrics/dataset_images.csv      one row per image file
  artifacts/metrics/dataset_instances.csv   one row per annotated person
  artifacts/reports/dataset_audit.md        human-readable audit
  artifacts/figures/dataset_*.png|svg       figures
  data/splits/cv_splits.json                repeated stratified group k-fold
  data/splits/instance_groups.csv           leakage groups per instance
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from salat_vision.config import get_config  # noqa: E402
from salat_vision.data import audit  # noqa: E402
from salat_vision.data.splits import make_cv_splits, save_splits  # noqa: E402


def main() -> int:
    cfg = get_config()
    idx, summary = audit.run_audit(cfg)
    paths = audit.save_tables(idx, summary, cfg)
    fig_dir = cfg.path("artifacts_dir") / "figures"
    li = idx.labeled_instances
    audit.fig_class_distribution(li, fig_dir / "dataset_class_distribution.png")
    audit.fig_source_by_class(li, idx.images, fig_dir / "dataset_source_by_class.png")
    audit.fig_box_geometry(li, fig_dir / "dataset_box_geometry.png")
    audit.fig_resolution(idx.images, fig_dir / "dataset_resolution.png")
    md = audit.write_markdown(summary, idx, cfg.path("artifacts_dir") / "reports" / "dataset_audit.md")

    sp = cfg["splits"]
    splits = make_cv_splits(li["label"].values, li["group_id"].values, n_splits=sp["n_splits"],
                            n_repeats=sp["n_repeats"], seed=cfg.seed)
    split_path = save_splits(li, splits, cfg.path("splits_dir"),
                             meta={"method": "StratifiedGroupKFold", "n_splits": sp["n_splits"], "n_repeats": sp["n_repeats"],
                                   "seed": cfg.seed, "n_instances": len(li), "n_groups": int(li["group_id"].nunique())})
    print(f"labeled instances: {len(li)} | groups: {li['group_id'].nunique()} | per class: {summary['instances_per_class']}")
    for k, p in {**paths, "report": md, "splits": split_path}.items():
        print(f"  {k:10s} -> {p.relative_to(cfg.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
