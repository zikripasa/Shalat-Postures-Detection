import numpy as np
import pytest

from salat_vision.inference.temporal import (STATES, TemporalConfig, decode_hmm, decode_raw, decode_smoothing, edit_score,
                                             segment_f1, summarize_segments, transition_matrix)
from salat_vision.labels import VISUAL_CLASSES

PLAN = [("qiyam", 4, "berdiri"), ("ruku", 2, "rukuk"), ("qiyam", 2, "iktidal"), ("sujud", 2, "sujud"),
        ("julus", 1.6, "duduk_antara_sujud"), ("sujud", 2, "sujud"), ("qiyam", 3, "berdiri"), ("ruku", 2, "rukuk"),
        ("qiyam", 2, "iktidal"), ("sujud", 2, "sujud"), ("julus", 1.6, "duduk_antara_sujud"), ("sujud", 2, "sujud"),
        ("julus", 5, "tasyahud")]


def synth(noise: float = 6.0, fps: float = 5.0, seed: int = 0, drop=()):
    rng = np.random.default_rng(seed)
    P, truth = [], []
    for vis, sec, phase in PLAN:
        for _ in range(int(round(sec * fps))):
            a = np.full(4, 0.15)
            a[VISUAL_CLASSES.index(vis)] = 0.55
            P.append(rng.dirichlet(a * noise))
            truth.append(phase)
    P = np.array(P)
    for i in drop:
        P[i] = np.nan
    return P, np.arange(len(P)) / fps, truth


@pytest.mark.parametrize("dt", [0.04, 0.2, 1.0])
def test_transition_matrix_is_stochastic(dt):
    A = transition_matrix(dt, TemporalConfig())
    assert A.shape == (len(STATES), len(STATES))
    np.testing.assert_allclose(A.sum(axis=1), 1.0)
    assert (A > 0).all(), "off-grammar jumps keep every path possible"


def test_hmm_recovers_two_rakaat_with_context_phases():
    P, t, truth = synth(seed=1, drop=(12, 13, 40))
    out = decode_hmm(P, t, TemporalConfig())
    acc = np.mean([a == b for a, b in zip(out["phase_per_frame"], truth)])
    assert acc > 0.9
    assert edit_score(out["phase_per_frame"], truth) == pytest.approx(100.0)
    summ = summarize_segments(out["segments"], video_seconds=t[-1] + 0.2)
    assert summ["sequence"][:4] == ["berdiri", "rukuk", "iktidal", "sujud"]
    assert summ["sequence"][-1] == "tasyahud"
    assert summ["cycles_observed"] == 2 and summ["rukuk_count"] == 2


def test_hmm_beats_raw_on_fragmentation():
    P, t, truth = synth(noise=3.0, seed=3)
    tc = TemporalConfig()
    raw = decode_raw(P, t, tc)
    hmm = decode_hmm(P, t, tc)
    assert len(hmm["segments"]) < len(raw["segments"])
    assert edit_score(hmm["phase_per_frame"], truth) > edit_score(raw["phase_per_frame"], truth)


def test_segments_are_contiguous_and_cover_video():
    P, t, _ = synth(seed=2)
    for dec in (decode_raw, decode_smoothing, decode_hmm):
        segs = dec(P, t, TemporalConfig())["segments"]
        assert segs[0]["start"] == 0.0
        for a, b in zip(segs, segs[1:]):
            assert a["end"] == pytest.approx(b["start"])
        assert sum(s["n_frames"] for s in segs) == len(P)


def test_low_confidence_segments_are_flagged_uncertain():
    P = np.full((25, 4), 0.25)
    t = np.arange(25) / 5
    segs = decode_hmm(P, t, TemporalConfig(uncertainty_threshold=0.55))["segments"]
    assert all(s["uncertain"] for s in segs)


def test_segment_f1_and_edit_score_bounds():
    a = ["berdiri"] * 5 + ["rukuk"] * 5
    assert segment_f1(a, a) == 100.0 and edit_score(a, a) == 100.0
    assert segment_f1(["sujud"] * 10, a) == 0.0
