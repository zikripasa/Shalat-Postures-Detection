"""Leakage-aware grouping and repeated stratified *group* k-fold splits.

Images are linked into groups when they are perceptual-hash near-duplicates or were
assigned to the same capture session during manual review
(``configs/session_groups.yaml``). Every instance inherits its image's group, and a
group never appears on both sides of a train/test (or train/validation) split.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

from salat_vision.config import Config, get_config


class _UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def load_session_groups(cfg: Config | None = None) -> dict[str, dict]:
    cfg = cfg or get_config()
    rel = cfg.get("splits", "session_groups_file")
    if not rel:
        return {}
    path = Path(rel) if Path(rel).is_absolute() else cfg.root / rel
    if not path.is_file():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("groups", {}) or {}


def assign_groups(images: pd.DataFrame, cfg: Config | None = None) -> tuple[pd.Series, pd.DataFrame]:
    """Return ``image_id -> group_id`` and a table of the links that created the groups."""
    cfg = cfg or get_config()
    ids = images["image_id"].tolist()
    uf = _UnionFind(ids)
    links: list[dict] = []

    threshold = int(cfg.get("splits", "phash_near_duplicate_distance", 18))
    hashes = {r.image_id: imagehash.hex_to_hash(r.phash) for r in images.itertuples() if isinstance(r.phash, str)}
    for a, b in itertools.combinations(sorted(hashes), 2):
        d = hashes[a] - hashes[b]
        if d <= threshold:
            uf.union(a, b)
            links.append({"image_a": a, "image_b": b, "reason": f"pHash distance {d} <= {threshold}"})

    for name, spec in load_session_groups(cfg).items():
        members = [m for m in spec.get("members", []) if m in uf.parent]
        for a, b in zip(members, members[1:]):
            uf.union(a, b)
            links.append({"image_a": a, "image_b": b, "reason": f"manual session group '{name}'"})

    roots = {i: uf.find(i) for i in ids}
    # stable, readable group ids: g000, g001, ... ordered by first member name
    ordered = sorted(set(roots.values()), key=lambda r: min(i for i in ids if roots[i] == r))
    names = {r: f"g{k:03d}" for k, r in enumerate(ordered)}
    groups = pd.Series({i: names[roots[i]] for i in ids}, name="group_id")
    return groups, pd.DataFrame(links)


def make_cv_splits(labels: np.ndarray, groups: np.ndarray, *, n_splits: int, n_repeats: int, seed: int) -> list[dict]:
    """Repeated stratified group k-fold over instances. Returns index arrays per fold."""
    splits: list[dict] = []
    n_groups = len(np.unique(groups))
    k = min(n_splits, n_groups)
    for r in range(n_repeats):
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
            assert not set(groups[tr]) & set(groups[te]), "group leakage between train and test"
            splits.append({"repeat": r, "fold": f, "train": tr.tolist(), "test": te.tolist()})
    return splits


def inner_train_val(train_idx: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Group-aware inner validation split taken from a training fold."""
    train_idx = np.asarray(train_idx)
    n_inner = max(2, int(round(1.0 / max(val_fraction, 1e-6))))
    n_inner = min(n_inner, len(np.unique(groups[train_idx])))
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros(len(train_idx)), labels[train_idx], groups[train_idx]))
    return train_idx[tr], train_idx[va]


def save_splits(instances: pd.DataFrame, splits: list[dict], out_dir: str | Path, meta: dict) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    instances[["instance_id", "image_id", "label", "group_id"]].to_csv(out_dir / "instance_groups.csv", index=False)
    payload = {
        "meta": meta,
        "instance_ids": instances["instance_id"].tolist(),
        "folds": [
            {"repeat": s["repeat"], "fold": s["fold"],
             "train": [instances["instance_id"].iloc[i] for i in s["train"]],
             "test": [instances["instance_id"].iloc[i] for i in s["test"]]}
            for s in splits
        ],
    }
    path = out_dir / "cv_splits.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
