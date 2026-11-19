#!/usr/bin/env python
"""Semi-synthetic benchmark of the temporal decoders.

No labelled prayer *videos* exist in the dataset, so sequences are simulated:

* Structure: 2-, 3- or 4-rakaat prayers (first tasyahud after rakaat 2 when n >= 3),
  phase durations drawn from plausible ranges, 0.4 s motion blends between postures and
  frames without a detected person (isolated drops + one burst per sequence).
* Emissions: real **out-of-fold** class probabilities of the selected model on held-out
  people (artifacts/metrics/oof_predictions.csv), so recognition errors are the model's
  real errors. Two noise models:
  - ``segment`` (conservative): one held-out prediction per segment, small per-frame
    jitter — errors persist across the segment as they would for one person in one pose.
  - ``iid`` (optimistic): a fresh held-out prediction for every frame.

Writes artifacts/metrics/temporal_benchmark.json (test seed 0) and, with ``--select``,
artifacts/metrics/temporal_decoder_selection.json (validation seed 1000).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from salat_vision.config import get_config  # noqa: E402
from salat_vision.inference.temporal import DECODERS, TemporalConfig, edit_score, segment_f1  # noqa: E402
from salat_vision.labels import PHASE_TO_VISUAL, VISUAL_CLASSES  # noqa: E402

RANGES = {"berdiri_first": (8, 20), "berdiri": (5, 15), "rukuk": (2, 5), "iktidal": (1.5, 4), "sujud": (2, 5),
          "duduk_antara_sujud": (1.5, 4), "tasyahud_awal": (6, 15), "tasyahud_akhir": (10, 25)}


def prayer_plan(n_rakaat: int, rng: np.random.Generator) -> list[tuple[str, float]]:
    plan = []
    for r in range(n_rakaat):
        plan.append(("berdiri", rng.uniform(*RANGES["berdiri_first" if r == 0 else "berdiri"])))
        for ph in ("rukuk", "iktidal", "sujud", "duduk_antara_sujud", "sujud"):
            plan.append((ph, rng.uniform(*RANGES[ph])))
        if r == 1 and n_rakaat >= 3:
            plan.append(("tasyahud", rng.uniform(*RANGES["tasyahud_awal"])))
    plan.append(("tasyahud", rng.uniform(*RANGES["tasyahud_akhir"])))
    return plan


def simulate(plan, pools, rng, *, fps: float, mode: str, blend_s: float = 0.4, drop_p: float = 0.03):
    seg_vecs = []
    for phase, _ in plan:
        pool = pools[PHASE_TO_VISUAL[phase]]
        seg_vecs.append(pool[rng.integers(len(pool))])
    P, truth = [], []
    for k, (phase, dur) in enumerate(plan):
        n = max(1, int(round(dur * fps)))
        for i in range(n):
            if mode == "iid":
                pool = pools[PHASE_TO_VISUAL[phase]]
                v = pool[rng.integers(len(pool))]
            else:
                v = rng.dirichlet(seg_vecs[k] * 30 + 0.05)
            # motion blend at the start of each segment (person moving from the previous posture)
            if k > 0 and i < blend_s * fps:
                w = (i + 1) / (blend_s * fps + 1)
                v = w * v + (1 - w) * seg_vecs[k - 1]
            P.append(v)
            truth.append(phase)
    P = np.array(P, dtype=float)
    T = len(P)
    drops = rng.random(T) < drop_p
    b = rng.integers(0, max(1, T - int(fps)))
    drops[b:b + int(fps)] = True  # one 1-second burst without a person
    P[drops] = np.nan
    return P, np.arange(T) / fps, truth


def evaluate(cfg, model_key: str, n_seq: int = 300, fps: float = 5.0, seed: int = 0) -> dict:
    oof = pd.read_csv(cfg.path("artifacts_dir") / "metrics" / "oof_predictions.csv")
    sel = oof[(oof["model"] == model_key) & (oof["view"] == "crop")]
    pcols = [f"p_{c}" for c in VISUAL_CLASSES]
    pools = {c: sel.loc[sel["true"] == c, pcols].to_numpy(dtype=float) for c in VISUAL_CLASSES}
    tc = TemporalConfig.from_cfg(cfg)
    results = {}
    for mode in ("segment", "iid"):
        rng = np.random.default_rng(seed)
        rows = []
        for s in range(n_seq):
            plan = prayer_plan(int(rng.choice([2, 3, 4])), rng)
            P, T, truth = simulate(plan, pools, rng, fps=fps, mode=mode)
            true_vis = [PHASE_TO_VISUAL[p] for p in truth]
            for name, dec in DECODERS.items():
                t0 = time.perf_counter()
                out = dec(P, T, tc)
                ms = (time.perf_counter() - t0) * 1000
                ph = out["phase_per_frame"]
                n_true_segs = len([1 for i in range(len(truth)) if i == 0 or truth[i] != truth[i - 1]])
                rows.append({
                    "seq": s, "decoder": name, "frames": len(T),
                    "phase_acc": float(np.mean([a == b for a, b in zip(ph, truth)])),
                    "visual_acc": float(np.mean([PHASE_TO_VISUAL[a] == b for a, b in zip(ph, true_vis)])),
                    "edit": edit_score(ph, truth),
                    "f1_10": segment_f1(ph, truth, 0.10), "f1_25": segment_f1(ph, truth, 0.25), "f1_50": segment_f1(ph, truth, 0.50),
                    "seg_ratio": len(out["segments"]) / n_true_segs,
                    "decode_ms": ms,
                })
        df = pd.DataFrame(rows)
        agg = df.groupby("decoder").agg(["mean", "std"])
        results[mode] = {
            dec: {m: {"mean": round(float(agg.loc[dec, (m, "mean")]), 4), "sd": round(float(agg.loc[dec, (m, "std")]), 4)}
                  for m in ("phase_acc", "visual_acc", "edit", "f1_10", "f1_25", "f1_50", "seg_ratio", "decode_ms")}
            for dec in DECODERS
        }
        results[mode]["n_sequences"] = n_seq
        results[mode]["mean_frames"] = float(df[df.decoder == "hmm"]["frames"].mean())
    return {
        "description": __doc__.strip(),
        "model_key": model_key,
        "sample_fps": fps,
        "pool_sizes": {c: int(len(v)) for c, v in pools.items()},
        "temporal_config": tc.__dict__,
        "results": results,
    }


def select_decoder(cfg, model_key: str, n_seq: int = 120, seed: int = 1000) -> dict:
    """Grid over decoder variants/settings on a *validation* seed (never the reported one)."""
    import itertools

    oof = pd.read_csv(cfg.path("artifacts_dir") / "metrics" / "oof_predictions.csv")
    sel = oof[(oof["model"] == model_key) & (oof["view"] == "crop")]
    pools = {c: sel.loc[sel["true"] == c, [f"p_{k}" for k in VISUAL_CLASSES]].to_numpy(float) for c in VISUAL_CLASSES}
    rows = []
    for mps, off in itertools.product([2.0, 3.0, 5.0], [0.02, 0.1, 0.3]):
        tc = TemporalConfig(mean_phase_seconds=mps, off_grammar_per_second=off)
        rng = np.random.default_rng(seed)
        acc = {d: [] for d in ("hmm", "hmm_local", "smoothing")}
        for _ in range(n_seq):
            P, T, truth = simulate(prayer_plan(int(rng.choice([2, 3, 4])), rng), pools, rng, fps=5.0, mode="segment")
            for d in acc:
                ph = DECODERS[d](P, T, tc)["phase_per_frame"]
                acc[d].append((np.mean([a == b for a, b in zip(ph, truth)]), segment_f1(ph, truth, 0.5) / 100))
        for d, v in acc.items():
            v = np.array(v)
            rows.append({"decoder": d, "mean_phase_seconds": mps, "off_grammar_per_second": off,
                         "phase_acc": round(float(v[:, 0].mean()), 4), "f1_50": round(float(v[:, 1].mean()), 4),
                         "score": round(float((v[:, 0].mean() + v[:, 1].mean()) / 2), 4)})
    rows.sort(key=lambda r: -r["score"])
    return {"criterion": "mean of frame phase accuracy and F1@50 on segment-consistent sequences",
            "validation_seed": seed, "n_sequences": n_seq, "best": rows[0], "grid": rows}


def main() -> int:
    cfg = get_config()
    if "--select" in sys.argv:
        selection = select_decoder(cfg, cfg["final_model"]["cv_model_key"])
        p = cfg.path("artifacts_dir") / "metrics" / "temporal_decoder_selection.json"
        p.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        print("validation-seed selection:", selection["best"], "->", p.relative_to(cfg.root))
    out = evaluate(cfg, cfg["final_model"]["cv_model_key"])
    path = cfg.path("artifacts_dir") / "metrics" / "temporal_benchmark.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["configured_decoder"] = cfg["temporal"].get("decoder", "hmm_local")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for mode, r in out["results"].items():
        print(f"== {mode} ({r['n_sequences']} sequences, ~{r['mean_frames']:.0f} frames each)")
        for dec in DECODERS:
            x = r[dec]
            print(f"  {dec:10s} phase acc {x['phase_acc']['mean']:.3f}  visual acc {x['visual_acc']['mean']:.3f}  "
                  f"edit {x['edit']['mean']:5.1f}  F1@50 {x['f1_50']['mean']:5.1f}  segs/true {x['seg_ratio']['mean']:.2f}  "
                  f"{x['decode_ms']['mean']:.1f} ms")
    print(f"-> {path.relative_to(cfg.root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
