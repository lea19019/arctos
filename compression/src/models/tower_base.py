"""TowerBase-7B-v0.1 loader (Unbabel; Llama-2 base + monolingual + bilingual CPT).

The pre-MT-SFT checkpoint of Tower. Same Llama-2 architecture as
TowerInstruct-7B-v0.2, just without the supervised MT fine-tuning step.
Used as the within-family ablation that isolates the effect of MT-SFT on
the depth signature for Q1/Q4.

Cite: Alves et al., COLM 2024 (Tower).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "Unbabel/TowerBase-7B-v0.1"


def load_tower_base(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
