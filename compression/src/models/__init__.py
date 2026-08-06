"""Model loaders for the three phase-one targets.

All three models are Llama-class decoder-only, so the TransformerLens API is
expected to apply uniformly. Per-model adaptations (Cohere attention scaling
on Aya, omt-llama tokenizer quirks, Tower's SFT prompt template) are
flagged inside each loader.

Tests live in `tests/models/`. CPU-tier tests verify the loader contract on
a tiny dummy; GPU-tier tests verify the real checkpoint loads and a single
forward pass returns the expected shape.
"""

from .aya import load_aya
from .omt_llama import load_omt_llama
from .tower import load_tower

__all__ = ["load_aya", "load_omt_llama", "load_tower"]
