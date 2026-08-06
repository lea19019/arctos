"""Tower-Plus-9B loader (Unbabel; Gemma 2 base + CPT + MT-SFT, 2025).

The successor to TowerInstruct. Despite being from the Tower line, the
underlying architecture is **Gemma 2** (Gemma2ForCausalLM), not Llama-2.
Gemma 2's module names (q_proj/k_proj/v_proj/o_proj + gate_proj/up_proj/
down_proj + RMSNorm) match LLAMA_PATHS exactly, so no new ArchPaths
needed.

Cite: Rei et al. (2025), "Tower+: Bridging Generality and Translation
Specialization in Multilingual LLMs", https://arxiv.org/abs/2506.17080
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "Unbabel/Tower-Plus-9B"


def load_tower_plus(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
