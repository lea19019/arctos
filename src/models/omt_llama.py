"""omt-llama-8b loader (Meta Omnilingual MT, Llama-class, ~1,600+ languages).

MT-purpose-built at extreme language scale. Decoder-only Llama-class.
Loaded via TransformerLens's general-llama path. The tokenizer carries
language tags; Q1 / Q2 prompts must respect the language-tag scheme or
the patching counterfactuals will not be meaningful.

Tests: `tests/models/test_loaders.py::test_omt_llama_*`.
"""

from __future__ import annotations

from typing import Any


HF_NAME = "facebook/omnilingual-mt-llama-8B"  # confirm exact HF id when first run


def load_omt_llama(*, dtype: str = "bfloat16", device: str = "cuda") -> Any:
    """Load omt-llama-8b via TransformerLens.

    See module docstring for tokenizer caveats.
    """
    import torch
    from transformer_lens import HookedTransformer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return HookedTransformer.from_pretrained(
        HF_NAME,
        dtype=torch_dtype,
        device=device,
    )
