"""Bounded-on-both-sides tests for src/rep_metrics.py (run: uv run pytest -q)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rep_metrics as rm  # noqa: E402


def test_no_repetition_is_clean():
    toks = list(range(200))
    assert rm.loop_onset(toks) == (None, None)
    assert rm.seq_rep_n(toks, 4) == 0.0
    assert rm.rep_r(toks) == 0.0


def test_we_we_we_loop_from_first_token_has_onset_below_t_min():
    # "We . We . We . ..." -> the 4-gram (We . We .) recurs immediately
    toks = ["We", "."] * 100
    t, gram = rm.loop_onset(toks)
    assert t is not None and t < rm.T_MIN
    assert gram == ("We", ".", "We", ".")
    s = rm.summarize(toks)
    assert s["loop"] and not s["loop_with_window"]
    assert s["seq_rep_4"] > 0.95
    assert s["rep_r"] == 1.0


def test_late_loop_has_window_and_correct_onset():
    clean = list(range(1000, 1060))          # 60 distinct tokens
    phrase = ["a", "b", "c", "d", "e"]
    toks = clean + phrase * 30              # loop begins at 60, 2nd occurrence at 65
    t, gram = rm.loop_onset(toks)
    assert t == 65                          # onset = second occurrence
    assert gram == tuple(phrase[:4])
    assert rm.summarize(toks)["loop_with_window"]


def test_two_occurrences_only_is_not_a_loop():
    # a 4-gram that appears exactly twice does not satisfy k=3
    toks = list(range(50)) + [1, 2, 3, 4] + list(range(100, 150)) + [1, 2, 3, 4] + list(range(200, 260))
    assert rm.loop_onset(toks) == (None, None)
    assert 0 < rm.seq_rep_n(toks, 4) < 0.1


def test_tng_source_control():
    src = ["x", "y", "z", "w"] * 3
    hyp_copy = src[:]                       # copying the source is not self-repetition
    assert rm.tng(hyp_copy, src) == 0
    hyp_loop = ["p", "q", "r", "s"] * 5
    assert rm.tng(hyp_loop, src) == 5 - 3


def test_short_sequences_do_not_crash():
    for toks in ([], [1], [1, 2, 3]):
        assert rm.loop_onset(toks) == (None, None)
        rm.summarize(toks)
