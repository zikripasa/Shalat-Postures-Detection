"""Posture classes, prayer phases and label normalisation.

Two label spaces are used deliberately:

* **Visual postures** (4 classes) — what a single image can show and what the
  dataset annotates: ``qiyam`` (standing), ``ruku`` (bowing), ``sujud``
  (prostration) and ``julus`` (sitting).
* **Prayer phases** (6 classes) — what the product reports for video. Two pairs
  of phases look the same in a single frame and are resolved from *temporal
  context* only (never from pixels):

  - Berdiri (qiyam) vs. Iktidal: both are standing; iktidal is the standing that
    directly follows a rukuk.
  - Duduk di antara dua sujud vs. Tasyahud: both are sitting; the first lies
    between two prostrations, tasyahud follows the second prostration.
"""

from __future__ import annotations

import re

#: Visual posture classes, in prayer order. Index order is the model output order.
VISUAL_CLASSES: list[str] = ["qiyam", "ruku", "sujud", "julus"]

VISUAL_LABELS_ID: dict[str, str] = {
    "qiyam": "Berdiri",
    "ruku": "Rukuk",
    "sujud": "Sujud",
    "julus": "Duduk",
}

VISUAL_LABELS_EN: dict[str, str] = {
    "qiyam": "Standing (qiyam)",
    "ruku": "Bowing (ruku)",
    "sujud": "Prostration (sujud)",
    "julus": "Sitting (julus)",
}

#: Prayer phases reported for video, in canonical order.
PHASES: list[str] = ["berdiri", "rukuk", "iktidal", "sujud", "duduk_antara_sujud", "tasyahud"]

PHASE_LABELS_ID: dict[str, str] = {
    "berdiri": "Berdiri (Qiyam)",
    "rukuk": "Rukuk",
    "iktidal": "Iktidal",
    "sujud": "Sujud",
    "duduk_antara_sujud": "Duduk di Antara Dua Sujud",
    "tasyahud": "Tasyahud",
}

PHASE_LABELS_EN: dict[str, str] = {
    "berdiri": "Standing (qiyam)",
    "rukuk": "Bowing (ruku)",
    "iktidal": "Standing after bowing (i'tidal)",
    "sujud": "Prostration (sujud)",
    "duduk_antara_sujud": "Sitting between prostrations",
    "tasyahud": "Testimony sitting (tasyahud)",
}

#: Which visual class each phase looks like in a single frame.
PHASE_TO_VISUAL: dict[str, str] = {
    "berdiri": "qiyam",
    "rukuk": "ruku",
    "iktidal": "qiyam",
    "sujud": "sujud",
    "duduk_antara_sujud": "julus",
    "tasyahud": "julus",
}

#: How each phase is identified: from pixels alone or with temporal context.
PHASE_EVIDENCE: dict[str, str] = {
    "berdiri": "visual + context",
    "rukuk": "visual",
    "iktidal": "visual + context",
    "sujud": "visual",
    "duduk_antara_sujud": "visual + context",
    "tasyahud": "visual + context",
}

UNCERTAIN = "tidak_pasti"
UNCERTAIN_LABEL_ID = "Tidak Pasti"

# Raw dataset labels -> visual class. The dataset README warns that 'sitting'/'julus'
# and 'sujud'/'Sujud' are spelling variants of the same class.
_RAW_LABEL_MAP: dict[str, str] = {
    "qiyam": "qiyam",
    "qiam": "qiyam",
    "qiyaam": "qiyam",
    "standing": "qiyam",
    "ruku": "ruku",
    "rukuk": "ruku",
    "ruku'": "ruku",
    "bowing": "ruku",
    "sujud": "sujud",
    "sujood": "sujud",
    "prostration": "sujud",
    "julus": "julus",
    "jalsa": "julus",
    "sitting": "julus",
}


def normalize_label(raw: str) -> str | None:
    """Map a raw annotation label onto a visual class (``None`` if unknown)."""
    if raw is None:
        return None
    key = re.sub(r"\s+", " ", str(raw)).strip().lower()
    return _RAW_LABEL_MAP.get(key)


def visual_index(label: str) -> int:
    return VISUAL_CLASSES.index(label)
