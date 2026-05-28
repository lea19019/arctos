"""Tests for src.interp.ifr.

CPU invariants:
- `IFRScores.layer_scores` has shape (L,); `head_scores` is (L, H);
  `mlp_scores` is (L,).
- Scores are non-negative.
- Per the L1-normalization step, summing the per-component contribution
  for any single token / position should be ~1.

GPU behavior:
- On a known calibration set, IFR layer ranking is stable across reruns
  (deterministic given the same seeded inputs).
- IFR's top-ranked layer agrees with activation patching's top-ranked layer
  to within a small tolerance on the same examples — this is the Q3
  cross-method validation step.

TODO: implement once `src.interp.ifr.ifr` is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.cpu
def test_score_shapes(tiny_model):
    """Layer / head / MLP score shapes match the model's architecture."""
    pytest.skip("TODO: implement after ifr is implemented.")


@pytest.mark.cpu
def test_scores_nonnegative(tiny_model):
    """All IFR scores >= 0."""
    pytest.skip("TODO: implement after ifr is implemented.")


@pytest.mark.cpu
def test_l1_normalization(tiny_model):
    """Per-token component contributions L1-normalize to ~1."""
    pytest.skip("TODO: implement after ifr is implemented.")


@pytest.mark.gpu
@pytest.mark.slow
def test_ifr_agrees_with_patching_on_top_layer(target_model):
    """IFR's top-ranked layer ≈ activation patching's top-ranked layer.

    This is the Q3 validation cross-check; it is allowed (and expected) to
    disagree sometimes — when it does, the question's notes.md should
    document the disagreement, not silence it.
    """
    pytest.skip("TODO: implement after ifr + activation_patching are implemented.")
