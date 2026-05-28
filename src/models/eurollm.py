"""EuroLLM-9B-Instruct loader (utter-project, 2024-25).

European multilingual specialist LLM, ~24 languages with strong focus
on EU official languages. Llama-class architecture, so LLAMA_PATHS
applies.

Gated on HF: requires access granted at
https://huggingface.co/utter-project/EuroLLM-9B-Instruct.

Cite: Martins et al. (2024), "EuroLLM: Multilingual Language Models for
Europe", https://arxiv.org/abs/2409.16235
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "utter-project/EuroLLM-9B-Instruct"


def load_eurollm(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
