"""BLOOM-7B1 loader (BigScience, 2022).

Old-generation multilingual decoder-only LM (46 natural languages + 13
programming languages). Uses **ALiBi** positional encoding (no RoPE),
**LayerNorm** (no RMSNorm), and the GPT-2-shaped module layout with
BLOOM-specific submodule names (handled by BLOOM_PATHS).

Cite: BigScience Workshop (2022), "BLOOM: A 176B-Parameter Open-Access
Multilingual Language Model", https://arxiv.org/abs/2211.05100

Note for quantization research: BLOOM is the "old generation" anchor
of the model set — pre-Llama-class, different positional encoding,
different normalization. If the quantization method generalizes here,
it generalizes far.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import BLOOM_PATHS, HookedModel


HF_NAME = "bigscience/bloom-7b1"


def load_bloom(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
    return HookedModel(hf_model, tokenizer, BLOOM_PATHS)
