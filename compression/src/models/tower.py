"""TowerInstruct-7B loader (Unbabel; Llama-2 base, mixed-monolingual+bilingual CPT then MT-task SFT).

MT-purpose-built at moderate scale (~10 languages). Standard Llama-2
architecture; loaded via HF + wrapped in `HookedModel`. The SFT prompt
template must be respected by `src/data/clean_corrupt.py`'s paired-prompt
construction — Tower has a specific instruction format and patching
experiments must respect it for the corrupt prompt to be a meaningful
counterfactual.

Cite: Alves et al., COLM 2024 (TowerInstruct).

Tests: `tests/models/test_loaders.py::test_tower_*`.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "Unbabel/TowerInstruct-7B-v0.2"


def load_tower(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
