"""Llama-3.1-8B-Instruct loader (Meta, mid-2024).

The most recent Meta decoder-only LM that fits a single A100 in bf16.
The Meta model WMT25 used in its constrained track. Standard Llama-3
architecture (RoPE, GQA, SwiGLU, RMSNorm) — module names match
LLAMA_PATHS.

Gated on HF: requires `huggingface-cli login` and access granted at
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "meta-llama/Llama-3.1-8B-Instruct"


def load_llama(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
