"""scikit-learn classifiers for pose features and frozen CNN embeddings.

Missing pose values are imputed with the training median and flagged with explicit
missing-value indicator columns (``SimpleImputer(add_indicator=True)``), so "the knee
was not visible" is itself information the model can use.

Hyper-parameters are chosen by an *inner* stratified group k-fold on the training fold
only (nested cross-validation), never on the test fold.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MACRO_F1 = make_scorer(f1_score, average="macro", zero_division=0)


def _pose_pipe(clf) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", clf),
    ])


def pose_model_specs(seed: int) -> dict[str, tuple[Pipeline, dict]]:
    """name -> (pipeline, param grid)."""
    return {
        "pose_logreg": (_pose_pipe(LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
                        {"clf__C": [0.1, 1.0, 10.0]}),
        "pose_svm_rbf": (_pose_pipe(SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
                         {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.05]}),
        "pose_random_forest": (_pose_pipe(RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                                                 random_state=seed, n_jobs=1)),
                               {"clf__max_depth": [None, 4], "clf__min_samples_leaf": [1, 2]}),
        "pose_mlp": (_pose_pipe(MLPClassifier(hidden_layer_sizes=(32,), max_iter=3000, random_state=seed)),
                     {"clf__alpha": [0.01, 0.1, 1.0]}),
    }


def probe_spec(seed: int) -> tuple[Pipeline, dict]:
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
    ])
    return pipe, {"clf__C": [0.01, 0.1, 1.0]}


def fit_with_inner_cv(pipe: Pipeline, grid: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *,
                      seed: int, n_inner: int = 3) -> tuple[Pipeline, dict]:
    """Grid-search on an inner stratified group k-fold (falls back to defaults if impossible)."""
    n_groups = len(np.unique(groups))
    min_class_groups = min(len(np.unique(groups[y == c])) for c in np.unique(y))
    k = min(n_inner, n_groups, max(min_class_groups, 2))
    if k < 2 or not grid:
        model = clone(pipe).fit(X, y)
        return model, {}
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs = GridSearchCV(clone(pipe), grid, scoring=MACRO_F1, cv=cv, refit=True, n_jobs=1, error_score=0.0)
        gs.fit(X, y, groups=groups)
    return gs.best_estimator_, gs.best_params_


def predict_proba_full(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """``predict_proba`` re-indexed to all classes (a fold may lack a class in training)."""
    p = model.predict_proba(X)
    classes = model.classes_ if hasattr(model, "classes_") else model[-1].classes_
    out = np.zeros((len(X), n_classes), dtype=np.float64)
    out[:, np.asarray(classes, dtype=int)] = p
    return out
