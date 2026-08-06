"""Aya Expanse 8B loader (Cohere; decoder-only, multilingual-by-pretraining).

Continuity model — the prior paper's target. Loaded via HuggingFace
transformers directly (TransformerLens has no Cohere/Aya support in 3.x);
the resulting model is wrapped in `HookedModel` (src.models._hooked) so
the interpretability methods see a uniform interface.

Aya Expanse is Cohere's CommandR/Cohere2 architecture: Llama-class decoder
with a custom attention scaling (1 / sqrt(d_head) is multiplied by an
extra factor inside the HF CohereAttention). For phase-one Q1 we don't
need to special-case it — the residual stream, MLP outputs, and W_O all
have the same shapes as Llama-2 — but Q2/Q3 (activation patching) should
sanity-check on a known induction head before trusting per-head numbers.

Tests: `tests/models/test_loaders.py::test_aya_*`.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel
from ._prompt import PAIR_TO_LANG_NAMES, build_mt_prompt, tokenize_target_prefix

__all__ = ["load_aya", "build_mt_prompt", "tokenize_target_prefix", "PAIR_TO_LANG_NAMES"]


HF_NAME = "CohereForAI/aya-expanse-8b"


def load_aya(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
    """Load Aya Expanse 8B and wrap it for interp.

    Args:
        dtype: model dtype; bfloat16 recommended on A100.
        device: cuda or cpu.

    Returns:
        A `HookedModel` exposing the TL-shaped API used by src.interp.*.
    """
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_NAME,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    hf_model.eval()
    # Cohere's HF config uses Llama-style attribute names (num_attention_heads,
    # hidden_size, num_hidden_layers) so LLAMA_PATHS works unchanged.
    return HookedModel(hf_model, tokenizer, LLAMA_PATHS)


# `build_mt_prompt` and `tokenize_target_prefix` are re-exported from
# `src.models._prompt` so all model loaders (Tower-Plus, BLOOM, Llama,
# EuroLLM) share the same generic MT prompt format — apples-to-apples
# across models. See _prompt.py for the rationale.


