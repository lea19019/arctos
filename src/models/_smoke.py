"""Tiny smoke-test loaders (NOT study models).

bloom-560m fits on the login-node T4 and shares the BLOOM architecture path,
so it validates the q6 pipeline end-to-end before the A100 sweep.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import BLOOM_PATHS, HookedModel

__all__ = ["load_bloom560m"]


def load_bloom560m(*, dtype: str = "float16", device: str = "cuda") -> HookedModel:
    torch_dtype = {"float32": torch.float32, "float16": torch.float16,
                   "bfloat16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
    hf = AutoModelForCausalLM.from_pretrained(
        "bigscience/bloom-560m", torch_dtype=torch_dtype, device_map=device)
    hf.eval()
    return HookedModel(hf, tok, BLOOM_PATHS)
