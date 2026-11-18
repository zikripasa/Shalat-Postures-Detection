#!/usr/bin/env python
"""Run the full cross-validated model comparison (all models, one protocol).

    python scripts/run_experiments.py                 # all models (fine-tuning takes ~30 min on 4 CPU cores)
    python scripts/run_experiments.py --skip-finetune # classical + probe models only (~3 min)

Writes artifacts/metrics/{cv_results.json, oof_predictions.csv, training_histories.json,
localization_eval.csv, model_comparison.csv}.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from salat_vision.config import get_config  # noqa: E402
from salat_vision.experiments import MODEL_INFO, build_feature_cache, run_cv, save_cv, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = get_config()
    models = args.models or [m for m in MODEL_INFO if not (args.skip_finetune and m.startswith("cnn_ft_"))]
    t0 = time.perf_counter()
    cache = build_feature_cache(cfg)
    cv = run_cv(cache, cfg, models=models, checkpoint_dir=cfg.path("checkpoints_dir"), log=lambda *a: print(*a, flush=True))
    summary = summarize(cache, cv, n_boot=args.n_boot, seed=cfg.seed)
    out_dir = cfg.path("artifacts_dir") / "metrics"
    paths = save_cv(cache, cv, summary, out_dir)
    rows = []
    for m, e in summary.items():
        rows.append({
            "model": m, "family": e["family"],
            "crop_macro_f1": e["crop"]["macro_f1_mean"], "crop_macro_f1_sd": e["crop"]["macro_f1_std"],
            "crop_macro_f1_ci_lo": e["crop"]["macro_f1_ci95"][0], "crop_macro_f1_ci_hi": e["crop"]["macro_f1_ci95"][1],
            "crop_accuracy": e["crop"]["accuracy_mean"],
            "image_macro_f1": e["image"]["macro_f1_mean"], "image_macro_f1_sd": e["image"]["macro_f1_std"],
            "image_macro_f1_ci_lo": e["image"]["macro_f1_ci95"][0], "image_macro_f1_ci_hi": e["image"]["macro_f1_ci95"][1],
            "image_accuracy": e["image"]["accuracy_mean"],
            "abstain_rate": e["crop"]["pooled"]["abstain_rate"],
            "ece": (e["crop"]["calibration"] or {}).get("ece"),
            "train_seconds_total": e["train_seconds_total"],
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "model_comparison.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"total {time.perf_counter() - t0:.0f}s")
    for k, p in paths.items():
        print(f"  {k}: {p.relative_to(cfg.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
