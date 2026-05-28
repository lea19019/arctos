"""Tests for src.interp.logit_lens.

CPU invariants:
- The logit-lens output at the LAST layer must equal the model's actual
  logits (up to fp tolerance) — a math-level sanity check.
- `LogitLensResult.layer_logits` has shape (L, V) for the requested position.
- When `target_tokens` is given, `target_token_mass` has shape (L, K) and
  every entry is in [0, 1].

GPU behavior:
- On a real MT prompt, target-language token mass should be ~zero in the
  early layers and rise toward the end — a behavioral sanity check that
  catches grossly wrong implementations.
"""

from __future__ import annotations

import pytest
import torch

from src.interp.logit_lens import logit_lens


@pytest.mark.cpu
def test_last_layer_matches_model_logits(tiny_model):
    """At the final layer, logit lens == actual model logits (within fp tol)."""
    prompt = "The capital of France is"
    result = logit_lens(tiny_model, prompt)
    with torch.no_grad():
        real_logits = tiny_model(tiny_model.to_tokens(prompt))[0, -1]
    diff = (real_logits.float() - result.layer_logits[-1].float()).abs().max().item()
    assert diff < 1e-4, f"last-layer lens vs real logits diff={diff}"


@pytest.mark.cpu
def test_shape_contract(tiny_model):
    """layer_logits is (L, V); target_token_mass is (L, K) when requested."""
    target_ids = tiny_model.tokenizer(" Paris", add_special_tokens=False)["input_ids"]
    assert len(target_ids) > 0, "tokenizer returned no tokens for ' Paris'"
    result = logit_lens(tiny_model, "The capital of France is", target_tokens=target_ids)
    assert result.layer_logits.shape == (tiny_model.cfg.n_layers, tiny_model.cfg.d_vocab)
    assert result.target_token_mass.shape == (tiny_model.cfg.n_layers, len(target_ids))


@pytest.mark.cpu
def test_target_mass_in_unit_interval(tiny_model):
    """All target_token_mass entries in [0, 1]."""
    target_ids = tiny_model.tokenizer(" Paris France London", add_special_tokens=False)["input_ids"]
    result = logit_lens(tiny_model, "The capital of France is", target_tokens=target_ids)
    mass = result.target_token_mass
    assert (mass >= 0).all() and (mass <= 1).all(), f"out-of-range mass: {mass}"


@pytest.mark.gpu
def test_target_language_emergence(target_model):
    """Target-language mass rises monotonically (in expectation) over depth.

    Behavioral check: on a real cs→de MT prompt, the German-token mass
    averaged over the gold target prefix should be lower in the first
    quartile of layers than in the last quartile. Not a strict per-layer
    monotonicity claim — that's what Q1 is investigating.
    """
    pytest.skip("TODO: implement after Q1 calibration data is loaded on GPU.")
