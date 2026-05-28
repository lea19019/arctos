"""TowerInstruct-7B loader (Unbabel; Llama-2 base, mixed-monolingual+bilingual CPT then MT-task SFT).

MT-purpose-built at moderate scale (~10 languages). Standard Llama-2
architecture, so TransformerLens drops in cleanly. The SFT prompt template
must be respected by `src/data/clean_corrupt.py`'s paired-prompt
construction — Tower has a specific instruction format and patching
experiments must respect it for the corrupt prompt to be a meaningful
counterfactual.

Cite: Alves et al., COLM 2024 (TowerInstruct).

Tests: `tests/models/test_loaders.py::test_tower_*`.
"""

from __future__ import annotations

from typing import Any


HF_NAME = "Unbabel/TowerInstruct-7B-v0.2"


def load_tower(*, dtype: str = "bfloat16", device: str = "cuda") -> Any:
    """Load TowerInstruct-7B via TransformerLens."""
    import torch
    from transformer_lens import HookedTransformer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return HookedTransformer.from_pretrained(
        HF_NAME,
        dtype=torch_dtype,
        device=device,
    )
