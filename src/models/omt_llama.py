"""omt-llama-8b loader (Meta Omnilingual MT, Llama-class, ~1,600+ languages).

MT-purpose-built at extreme language scale. Decoder-only Llama-class.
Loaded via HF + wrapped in `HookedModel`. The tokenizer carries language
tags; Q1 / Q2 prompts must respect the language-tag scheme or the
patching counterfactuals will not be meaningful.

Tests: `tests/models/test_loaders.py::test_omt_llama_*`.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "facebook/omnilingual-mt-llama-8B"  # confirm exact HF id when first run


def load_omt_llama(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_NAME, torch_dtype=torch_dtype, device_map=device
    )
    hf_model.eval()
    return HookedModel(hf_model, tokenizer, LLAMA_PATHS)
