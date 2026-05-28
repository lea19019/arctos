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

TODO: implement once `src.interp.logit_lens.logit_lens` is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.cpu
def test_last_layer_matches_model_logits(tiny_model):
    """At the final layer, logit lens == actual model logits (within fp tol)."""
    pytest.skip("TODO: implement after logit_lens is implemented.")


@pytest.mark.cpu
def test_shape_contract(tiny_model):
    """layer_logits is (L, V); target_token_mass is (L, K) when requested."""
    pytest.skip("TODO: implement after logit_lens is implemented.")


@pytest.mark.cpu
def test_target_mass_in_unit_interval(tiny_model):
    """All target_token_mass entries in [0, 1]."""
    pytest.skip("TODO: implement after logit_lens is implemented.")


@pytest.mark.gpu
def test_target_language_emergence(target_model):
    """Target-language mass rises monotonically (in expectation) over depth.

    Behavioral check: on a real cs→de MT prompt, the German-token mass
    averaged over the gold target prefix should be lower in the first
    quartile of layers than in the last quartile. Not a strict per-layer
    monotonicity claim — that's what Q1 is investigating.
    """
    pytest.skip("TODO: implement after logit_lens + Q1 calibration data exist.")
