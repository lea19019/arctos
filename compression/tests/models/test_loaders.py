"""Tests for src.models loaders.

CPU invariants:
- Each loader function exists and accepts (dtype, device); when called with
  device="cpu" it should at least *try* to dispatch to CPU rather than
  hard-coding cuda.

GPU behavior:
- The model loads on cuda; a single forward pass on a 1-token input returns
  logits with shape (1, V).
- The TransformerLens config's attention scaling matches expectation —
  Cohere on Aya, standard on Tower / omt-llama.

TODO: implement once each loader is implemented.
"""

from __future__ import annotations

import pytest


@pytest.mark.cpu
def test_load_aya_signature():
    pytest.skip("TODO: implement after src.models.aya.load_aya is implemented.")


@pytest.mark.cpu
def test_load_omt_llama_signature():
    pytest.skip("TODO: implement after src.models.omt_llama.load_omt_llama is implemented.")


@pytest.mark.cpu
def test_load_tower_signature():
    pytest.skip("TODO: implement after src.models.tower.load_tower is implemented.")


@pytest.mark.gpu
def test_aya_forward_pass(target_model):
    """Real Aya checkpoint loads and a 1-token forward returns (1, V) logits.

    Also verifies Cohere attention scaling matches expectation — the lens
    sanity check from the loader's docstring.
    """
    pytest.skip("TODO: implement after src.models.aya.load_aya is implemented.")


@pytest.mark.gpu
def test_omt_llama_forward_pass(target_model):
    pytest.skip("TODO: implement after src.models.omt_llama.load_omt_llama is implemented.")


@pytest.mark.gpu
def test_tower_forward_pass(target_model):
    pytest.skip("TODO: implement after src.models.tower.load_tower is implemented.")
