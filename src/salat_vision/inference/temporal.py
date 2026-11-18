"""Temporal decoding: frame probabilities -> prayer-phase segments.

The posture model only sees **4 visual classes**. The product reports **6 phases**, two
pairs of which are visually identical and can only be told apart by *order*:

    Berdiri -> Rukuk -> Iktidal -> Sujud -> Duduk antara sujud -> Sujud -> (Berdiri | Tasyahud)

``decode_hmm`` runs Viterbi over a 7-state hidden Markov model whose transitions follow
this grammar (two sujud states share one visual class) with a small probability of
off-grammar jumps, so videos that start mid-prayer or contain recognition errors still
decode. Emissions are the classifier's frame probabilities for each state's visual class;
frames without a detected person are uninformative (uniform emission).

``decode_smoothing`` (moving average + minimum duration + rule-based context labels) and
``decode_raw`` (per-frame argmax + the same context rules) are simpler baselines.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from salat_vision.labels import PHASE_EVIDENCE, PHASE_LABELS_ID, PHASE_TO_VISUAL, PHASES, UNCERTAIN, UNCERTAIN_LABEL_ID, VISUAL_CLASSES

# ---------------------------------------------------------------- hidden states & grammar
STATES = ["berdiri", "rukuk", "iktidal", "sujud_1", "duduk_antara_sujud", "sujud_2", "tasyahud"]
STATE_PHASE = {"berdiri": "berdiri", "rukuk": "rukuk", "iktidal": "iktidal", "sujud_1": "sujud",
               "duduk_antara_sujud": "duduk_antara_sujud", "sujud_2": "sujud", "tasyahud": "tasyahud"}
STATE_VISUAL = {s: PHASE_TO_VISUAL[STATE_PHASE[s]] for s in STATES}
GRAMMAR = {
    "berdiri": ["rukuk"],
    "rukuk": ["iktidal"],
    "iktidal": ["sujud_1"],
    "sujud_1": ["duduk_antara_sujud"],
    "duduk_antara_sujud": ["sujud_2"],
    "sujud_2": ["berdiri", "tasyahud"],
    "tasyahud": ["berdiri"],
}
S = len(STATES)
_SI = {s: i for i, s in enumerate(STATES)}
_VIS = np.array([VISUAL_CLASSES.index(STATE_VISUAL[s]) for s in STATES])


@dataclass
class TemporalConfig:
    mean_phase_seconds: float = 3.0      # sets the per-frame switch probability
    off_grammar_per_second: float = 0.02  # total probability mass per second for off-grammar jumps
    emission_power: float = 1.0          # <1 softens over-confident frame probabilities
    min_prob: float = 1e-3               # floor on emissions (keeps decoding robust to 0/1 outputs)
    uncertainty_threshold: float = 0.55
    min_segment_seconds: float = 0.6
    smoothing_seconds: float = 1.0       # window of the moving-average baseline

    @classmethod
    def from_cfg(cls, cfg) -> "TemporalConfig":
        t = cfg["temporal"] if cfg is not None else {}
        inf = cfg["inference"] if cfg is not None else {}
        return cls(
            mean_phase_seconds=float(t.get("mean_phase_seconds", 3.0)),
            off_grammar_per_second=float(t.get("off_grammar_per_second", 0.02)),
            emission_power=float(t.get("emission_power", 1.0)),
            uncertainty_threshold=float(inf.get("uncertainty_threshold", 0.55)),
            min_segment_seconds=float(t.get("min_segment_seconds", 0.6)),
            smoothing_seconds=float(t.get("smoothing_seconds", 1.0)),
        )


def transition_matrix(dt: float, tc: TemporalConfig) -> np.ndarray:
    """Row-stochastic 7x7 transition matrix for a frame interval of ``dt`` seconds."""
    p_switch = 1.0 - math.exp(-dt / max(tc.mean_phase_seconds, 1e-3))
    p_off = min(1.0 - math.exp(-dt * tc.off_grammar_per_second), p_switch * 0.5)
    A = np.zeros((S, S))
    for s, nxt in GRAMMAR.items():
        i = _SI[s]
        off = [j for j in range(S) if j != i and STATES[j] not in nxt]
        A[i, i] = 1.0 - p_switch
        for n in nxt:
            A[i, _SI[n]] = (p_switch - p_off) / len(nxt)
        for j in off:
            A[i, j] = p_off / len(off)
    return A


def initial_distribution() -> np.ndarray:
    pi = np.full(S, 0.6 / (S - 1))
    pi[_SI["berdiri"]] = 0.4  # most recordings start standing, but any start is allowed
    return pi / pi.sum()


def _emissions(P: np.ndarray, tc: TemporalConfig) -> np.ndarray:
    """(T, 4) visual probabilities (NaN rows = no person) -> (T, 7) emission scores."""
    T = len(P)
    E = np.ones((T, S))
    ok = ~np.isnan(P).any(axis=1)
    if ok.any():
        Q = np.clip(P[ok], tc.min_prob, 1.0) ** tc.emission_power
        Q = Q / Q.sum(axis=1, keepdims=True)
        E[ok] = Q[:, _VIS]
    return E


def viterbi(P: np.ndarray, times: np.ndarray, tc: TemporalConfig) -> np.ndarray:
    T = len(P)
    if T == 0:
        return np.zeros(0, dtype=int)
    logE = np.log(_emissions(P, tc))
    delta = np.log(initial_distribution()) + logE[0]
    back = np.zeros((T, S), dtype=int)
    for t in range(1, T):
        dt = max(float(times[t] - times[t - 1]), 1e-3)
        logA = np.log(transition_matrix(dt, tc))
        cand = delta[:, None] + logA  # from i to j
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + logE[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def forward_backward(P: np.ndarray, times: np.ndarray, tc: TemporalConfig) -> np.ndarray:
    """Posterior state marginals (T, 7) — used for smoothed visual probabilities."""
    T = len(P)
    E = _emissions(P, tc)
    alpha = np.zeros((T, S))
    c = np.zeros(T)
    a = initial_distribution() * E[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    As = [None] * T
    for t in range(1, T):
        As[t] = transition_matrix(max(float(times[t] - times[t - 1]), 1e-3), tc)
        a = (alpha[t - 1] @ As[t]) * E[t]
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.ones((T, S))
    for t in range(T - 2, -1, -1):
        b = As[t + 1] @ (E[t + 1] * beta[t + 1])
        beta[t] = b / c[t + 1]
    g = alpha * beta
    return g / g.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- segments
def _runs(labels: list) -> list[tuple[int, int, object]]:
    """[(start_idx, end_idx_inclusive, label), ...] for consecutive equal labels."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((start, i - 1, labels[start]))
            start = i
    return out


def _segment_records(phase_per_frame: list[str], P: np.ndarray, times: np.ndarray, frame_dt: float,
                     tc: TemporalConfig, state_per_frame: list[str] | None = None) -> list[dict]:
    segs = []
    for k, (a, b, phase) in enumerate(_runs(phase_per_frame)):
        start = float(times[a])
        end = float(times[b] + frame_dt)
        vis = PHASE_TO_VISUAL[phase]
        vi = VISUAL_CLASSES.index(vis)
        block = P[a:b + 1]
        ok = ~np.isnan(block).any(axis=1)
        conf = float(block[ok, vi].mean()) if ok.any() else 0.0
        no_person = float((~ok).mean())
        uncertain = conf < tc.uncertainty_threshold or no_person > 0.5
        segs.append({
            "index": k,
            "phase": phase,
            "label": PHASE_LABELS_ID[phase],
            "visual_class": vis,
            "evidence": PHASE_EVIDENCE[phase],
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "n_frames": int(b - a + 1),
            "confidence": round(conf, 4),
            "frames_without_person": round(no_person, 4),
            "uncertain": bool(uncertain),
            "display_label": UNCERTAIN_LABEL_ID if uncertain else PHASE_LABELS_ID[phase],
            "state": state_per_frame[a] if state_per_frame else None,
        })
    return segs


def _context_phases(visual: list[str | None]) -> list[str]:
    """Rule-based phase labels from a sequence of visual-class *segments* (baselines)."""
    runs = _runs(visual)
    phase_of_run = []
    for k, (_, _, v) in enumerate(runs):
        prev = next((runs[j][2] for j in range(k - 1, -1, -1) if runs[j][2] is not None), None)
        nxt = next((runs[j][2] for j in range(k + 1, len(runs)) if runs[j][2] is not None), None)
        if v is None:
            phase_of_run.append(phase_of_run[-1] if phase_of_run else "berdiri")
        elif v == "qiyam":
            phase_of_run.append("iktidal" if prev == "ruku" else "berdiri")
        elif v == "ruku":
            phase_of_run.append("rukuk")
        elif v == "sujud":
            phase_of_run.append("sujud")
        else:  # julus
            phase_of_run.append("duduk_antara_sujud" if (prev == "sujud" and nxt == "sujud") else "tasyahud")
    out: list[str] = []
    for (a, b, _), ph in zip(runs, phase_of_run):
        out += [ph] * (b - a + 1)
    return out


def _merge_short(visual: list, times: np.ndarray, min_seconds: float, frame_dt: float) -> list:
    """Absorb runs shorter than ``min_seconds`` into their longer neighbour."""
    vis = list(visual)
    changed = True
    while changed:
        changed = False
        runs = _runs(vis)
        if len(runs) <= 1:
            break
        for k, (a, b, v) in enumerate(runs):
            dur = times[b] - times[a] + frame_dt
            if dur < min_seconds:
                left = runs[k - 1] if k > 0 else None
                right = runs[k + 1] if k + 1 < len(runs) else None
                cands = [r for r in (left, right) if r is not None]
                tgt = max(cands, key=lambda r: r[1] - r[0])
                for i in range(a, b + 1):
                    vis[i] = tgt[2]
                changed = True
                break
    return vis


def _frame_dt(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 0.2


def decode_raw(P: np.ndarray, times: np.ndarray, tc: TemporalConfig) -> dict:
    ok = ~np.isnan(P).any(axis=1)
    visual = [VISUAL_CLASSES[int(P[t].argmax())] if ok[t] else None for t in range(len(P))]
    phases = _context_phases(visual)
    return {"method": "raw", "phase_per_frame": phases,
            "segments": _segment_records(phases, P, times, _frame_dt(times), tc)}


def decode_smoothing(P: np.ndarray, times: np.ndarray, tc: TemporalConfig) -> dict:
    dt = _frame_dt(times)
    w = max(1, int(round(tc.smoothing_seconds / dt)))
    Pf = np.where(np.isnan(P), np.nan, P)
    Q = np.full_like(P, np.nan)
    for t in range(len(P)):
        lo, hi = max(0, t - w // 2), min(len(P), t + w // 2 + 1)
        block = Pf[lo:hi]
        okb = ~np.isnan(block).any(axis=1)
        if okb.any():
            Q[t] = block[okb].mean(axis=0)
    ok = ~np.isnan(Q).any(axis=1)
    visual = [VISUAL_CLASSES[int(Q[t].argmax())] if ok[t] else None for t in range(len(Q))]
    # forward-fill gaps so they do not create spurious segments
    last = None
    for t in range(len(visual)):
        if visual[t] is None:
            visual[t] = last
        else:
            last = visual[t]
    visual = _merge_short(visual, times, tc.min_segment_seconds, dt)
    phases = _context_phases(visual)
    return {"method": "smoothing", "phase_per_frame": phases, "smoothed": Q,
            "segments": _segment_records(phases, P, times, dt, tc)}


def decode_hmm(P: np.ndarray, times: np.ndarray, tc: TemporalConfig) -> dict:
    path = viterbi(P, times, tc)
    states = [STATES[i] for i in path]
    phases = [STATE_PHASE[s] for s in states]
    post = forward_backward(P, times, tc) if len(P) else np.zeros((0, S))
    smoothed = np.zeros((len(P), len(VISUAL_CLASSES)))
    for si in range(S):
        smoothed[:, _VIS[si]] += post[:, si]
    return {"method": "hmm", "phase_per_frame": phases, "state_per_frame": states, "smoothed": smoothed,
            "segments": _segment_records(phases, P, times, _frame_dt(times), tc, states)}


DECODERS = {"raw": decode_raw, "smoothing": decode_smoothing, "hmm": decode_hmm}


def summarize_segments(segments: list[dict], *, video_seconds: float) -> dict:
    """Durations per phase, uncertain list, and observed counts (descriptive only)."""
    dur = {p: 0.0 for p in PHASES}
    for s in segments:
        dur[s["phase"]] += s["duration"]
    uncertain = [s for s in segments if s["uncertain"]]
    states = [s.get("state") for s in segments]
    return {
        "phase_seconds": {p: round(v, 3) for p, v in dur.items()},
        "phase_share": {p: round(v / video_seconds, 4) if video_seconds > 0 else 0.0 for p, v in dur.items()},
        "n_segments": len(segments),
        "n_uncertain_segments": len(uncertain),
        "uncertain_seconds": round(sum(s["duration"] for s in uncertain), 3),
        "sequence": [s["phase"] for s in segments],
        "rukuk_count": sum(1 for s in segments if s["phase"] == "rukuk"),
        "sujud_count": sum(1 for s in segments if s["phase"] == "sujud"),
        # every completed cycle passes through the second prostration exactly once
        "cycles_observed": sum(1 for st in states if st == "sujud_2") if any(states) else None,
        "uncertain_label": UNCERTAIN,
    }


# ---------------------------------------------------------------- evaluation helpers
def levenshtein(a: list, b: list) -> int:
    dp = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[-1]


def edit_score(pred_phases: list[str], true_phases: list[str]) -> float:
    """Segmental edit score (0-100) used in temporal action segmentation."""
    p = [r[2] for r in _runs(pred_phases)]
    t = [r[2] for r in _runs(true_phases)]
    return 100.0 * (1.0 - levenshtein(p, t) / max(len(p), len(t), 1))


def segment_f1(pred_phases: list[str], true_phases: list[str], iou_threshold: float = 0.5) -> float:
    """F1@IoU for segments (Lea et al.): a prediction is a TP if it overlaps an unmatched
    same-label ground-truth segment with IoU >= threshold."""
    P_ = _runs(pred_phases)
    T_ = _runs(true_phases)
    used = [False] * len(T_)
    tp = 0
    for a, b, lab in P_:
        best, best_j = 0.0, -1
        for j, (c, d, tl) in enumerate(T_):
            if tl != lab or used[j]:
                continue
            inter = max(0, min(b, d) - max(a, c) + 1)
            union = max(b, d) - min(a, c) + 1
            iou = inter / union
            if iou > best:
                best, best_j = iou, j
        if best >= iou_threshold:
            tp += 1
            used[best_j] = True
    fp = len(P_) - tp
    fn = len(T_) - tp
    return 100.0 * (2 * tp / max(2 * tp + fp + fn, 1))
