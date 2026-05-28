"""Tests for src.interp.ifr.

CPU invariants:
- `IFRScores.layer_scores` has shape (L,); `head_scores` is (L, H);
  `mlp_scores` is (L,).
- Scores are non-negative.
- Per the L1-normalization step, summing the per-component contribution
  for any single token / position should be ~1 (layer_scores + embed_score).

GPU behavior:
- On a known calibration set, IFR layer ranking is stable across reruns
  (deterministic given the same seeded inputs).
- IFR's top-ranked layer agrees with activation patching's top-ranked layer
  to within a small tolerance on the same examples — this is the Q3
  cross-method validation step.
"""

from __future__ import annotations

import pytest

from src.interp.ifr import ifr


@pytest.mark.cpu
def test_score_shapes(tiny_model):
    """Layer / head / MLP score shapes match the model's architecture."""
    scores = ifr(tiny_model, ["The capital of France is"], target_position="last")
    assert scores.layer_scores.shape == (tiny_model.cfg.n_layers,)
    assert scores.head_scores.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.n_heads)
    assert scores.mlp_scores.shape == (tiny_model.cfg.n_layers,)
    assert isinstance(scores.embed_score, float)
    assert scores.n_examples == 1


@pytest.mark.cpu
def test_scores_nonnegative(tiny_model):
    """All IFR scores >= 0."""
    scores = ifr(tiny_model, ["The capital of France is", "Cats sit on the mat."])
    assert (scores.layer_scores >= 0).all()
    assert (scores.head_scores >= 0).all()
    assert (scores.mlp_scores >= 0).all()
    assert scores.embed_score >= 0


@pytest.mark.cpu
def test_l1_normalization(tiny_model):
    """Per-example mass L1-normalizes to ~1 (so the average stays ~1)."""
    scores = ifr(tiny_model, ["The capital of France is", "Cats sit on the mat."])
    total = scores.layer_scores.sum().item() + scores.embed_score
    assert abs(total - 1.0) < 1e-3, f"total mass {total} != 1"


@pytest.mark.gpu
@pytest.mark.slow
def test_ifr_agrees_with_patching_on_top_layer(target_model):
    """IFR's top-ranked layer ≈ activation patching's top-ranked layer.

    This is the Q3 validation cross-check; it is allowed (and expected) to
    disagree sometimes — when it does, the question's notes.md should
    document the disagreement, not silence it.
    """
    pytest.skip("TODO: implement after activation_patching is implemented.")
