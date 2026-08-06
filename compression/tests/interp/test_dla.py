"""Tests for src.interp.dla.

CPU invariants:
- `DLAScores.layer_attn / layer_mlp / head_scores` shapes match the model.
- DLA score for a token the model is *confident* about should be positive
  on at least some components, negative on others (mixed sign — proves
  attribution is signed, unlike IFR).
- head_scores.sum(dim=-1) ≈ layer_attn (per-layer attn is the sum of heads).
"""

from __future__ import annotations

import pytest
import torch

from src.interp.dla import dla


@pytest.mark.cpu
def test_dla_shapes(tiny_model):
    tok_ids = tiny_model.tokenizer(" Paris", add_special_tokens=False)["input_ids"]
    scores = dla(tiny_model, [("The capital of France is", tok_ids)])
    L, H = tiny_model.cfg.n_layers, tiny_model.cfg.n_heads
    assert scores.layer_attn.shape == (L,)
    assert scores.layer_mlp.shape == (L,)
    assert scores.head_scores.shape == (L, H)
    assert scores.n_examples == 1


@pytest.mark.cpu
def test_layer_attn_is_head_sum(tiny_model):
    tok_ids = tiny_model.tokenizer(" Paris", add_special_tokens=False)["input_ids"]
    scores = dla(tiny_model, [("The capital of France is", tok_ids)])
    assert torch.allclose(scores.head_scores.sum(dim=-1), scores.layer_attn, atol=1e-3)


@pytest.mark.cpu
def test_mixed_sign_contributions(tiny_model):
    """DLA is signed — for any target, at least one (layer, head) should push toward
    it and one should push away. (Unlike IFR magnitudes which are non-negative.)"""
    tok_ids = tiny_model.tokenizer(" Paris", add_special_tokens=False)["input_ids"]
    scores = dla(tiny_model, [("The capital of France is", tok_ids)])
    assert (scores.head_scores > 0).any(), "no head pushes toward the target"
    assert (scores.head_scores < 0).any(), "no head pushes away from the target"
