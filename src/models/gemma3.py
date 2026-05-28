"""Gemma-3-12B-it loader (Google, 2025 — WMT25's Google entry).

**Baseline only.** Per the project's quantization pivot, we do not try to
beat Google's compression stack (MatFormer / PLE / QAT). Gemma 3 is here
to check whether our interpretability-guided findings *generalize* to a
Gemma-family architecture — which Tower-Plus (also Gemma-2-derived)
already hinted breaks the cross-architecture depth signature.

Notes:
- Gemma 4 (`model_type: gemma4`) is not supported by transformers 4.57.x;
  upgrading transformers would risk the pinned env every other model
  depends on. Gemma-3-12B is the version WMT25 actually used and loads
  cleanly here.
- Gemma3ForConditionalGeneration is a multimodal wrapper; the text
  decoder is nested under model.language_model (see GEMMA3_PATHS). We
  load via AutoModelForCausalLM and only ever run text input, so the
  vision tower is dead weight in memory but never executed.
- head_dim=256 is decoupled from hidden_size/n_heads — HookedModel reads
  cfg.head_dim, so the W_O reshape is correct.

Cite: Gemma Team (2025), "Gemma 3 Technical Report".
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import GEMMA3_PATHS, HookedModel


HF_NAME = "google/gemma-3-12b-it"


def load_gemma3(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
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
    return HookedModel(hf_model, tokenizer, GEMMA3_PATHS)
