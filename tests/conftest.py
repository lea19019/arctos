"""Pytest configuration.

- Auto-skips `@pytest.mark.gpu` tests when CUDA is unavailable.
- Provides a `tiny_model` fixture for CPU tests (gpt2 via TransformerLens).
- Provides a parameterized `target_model` fixture for GPU tests, iterating
  Aya / omt-llama / Tower.

The fixtures here are intentionally small — anything model-specific belongs
inside the test that uses it, not in the global conftest.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Skip @pytest.mark.gpu tests if CUDA is unavailable."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if cuda_available:
        return

    skip_gpu = pytest.mark.skip(reason="CUDA not available; skipping GPU-tier tests.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def tiny_model():
    """A tiny HookedModel for CPU shape / math invariant tests.

    Uses HF's gpt2 (~125M params; fits in CPU RAM, runs in seconds),
    wrapped by src.models._hooked.HookedModel via the GPT-2 path spec.
    Loaded once per session for speed.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.models._hooked import GPT2_PATHS, HookedModel

    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype="float32").to("cpu").eval()
    return HookedModel(hf, tok, GPT2_PATHS)


@pytest.fixture(scope="session", params=["aya", "omt_llama", "tower"])
def target_model(request):
    """A real target model loaded on GPU, parameterized over the three.

    GPU tests using this fixture run three times (once per model), unless
    the test narrows via `pytest.param(..., marks=...)` or by explicitly
    skipping models that don't exercise the behavior under test.

    TODO: wire to src.models.{aya,omt_llama,tower}.load_* once those
    loaders are implemented. Until then, every GPU test that requests this
    fixture is skipped.
    """
    pytest.skip(f"target_model[{request.param}]: implement loader first.")
