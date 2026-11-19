"""Classification metrics with honest uncertainty for small, grouped data."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

ABSTAIN = -1


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Metrics where ``y_pred == ABSTAIN`` counts as an error (never as a correct class)."""
    labels = list(range(len(class_names)))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, np.where(y_pred == ABSTAIN, len(labels), y_pred), labels=labels + [len(labels)])
    present = s > 0
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        # mean recall over classes present in y_true (= balanced accuracy; abstentions count as misses)
        "balanced_accuracy": float(np.mean(r[present])) if present.any() else float("nan"),
        "abstain_rate": float((y_pred == ABSTAIN).mean()),
        "per_class": {
            name: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, name in enumerate(class_names)
        },
        # rows = true class, columns = predicted class (+ final column "abstain")
        "confusion_matrix": cm[: len(labels)].tolist(),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def cluster_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray, metric, *, n_boot: int = 2000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI from resampling *groups* (not rows) with replacement.

    Rows from the same capture session are correlated; resampling groups keeps the
    interval honest about how few independent observations there really are.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in pick])
        stats.append(metric(y_true[idx], y_pred[idx]))
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 5) -> dict:
    """Top-label ECE with equal-width confidence bins (+ the reliability-diagram data)."""
    proba = np.asarray(proba, dtype=float)
    ok = ~np.isnan(proba).any(axis=1)
    y_true, proba = np.asarray(y_true)[ok], proba[ok]
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "accuracy": float(correct[m].mean()),
                         "confidence": float(conf[m].mean())})
        else:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "accuracy": None, "confidence": None})
    return {"ece": float(ece), "n": int(len(y_true)), "bins": bins}


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    proba = np.asarray(proba, dtype=float)
    ok = ~np.isnan(proba).any(axis=1)
    onehot = np.eye(proba.shape[1])[np.asarray(y_true)[ok]]
    return float(((proba[ok] - onehot) ** 2).sum(axis=1).mean())
