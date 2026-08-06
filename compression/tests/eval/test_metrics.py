"""Tests for src.eval.metrics.

CPU invariants:
- `evaluate(hyps, refs, pair=...)` returns a MetricScores with finite BLEU
  / chrF++ when given a non-empty hyp/ref pair.
- For en-zh, the sacrebleu tokenizer used must be `zh` (segmentation
  affects BLEU; getting this wrong silently produces meaningless numbers).

GPU optional — COMET runs on GPU when available; mark its test accordingly.

TODO: implement once src.eval.metrics.evaluate is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.cpu
def test_evaluate_returns_finite_bleu_chrf():
    pytest.skip("TODO: implement after evaluate is implemented.")


@pytest.mark.cpu
def test_zh_tokenizer_used_for_en_zh():
    """The pair='en-zh' branch must route to sacrebleu's `zh` tokenizer."""
    pytest.skip("TODO: implement after evaluate is implemented.")


@pytest.mark.gpu
def test_comet_score_finite():
    """COMET runs on GPU and returns a finite score."""
    pytest.skip("TODO: implement after evaluate's COMET path is implemented.")
