"""Aya Expanse 8B loader (Cohere; decoder-only, multilingual-by-pretraining).

Continuity model — the prior paper's target. Loaded via HuggingFace
transformers directly (TransformerLens has no Cohere/Aya support in 3.x);
the resulting model is wrapped in `HookedModel` (src.models._hooked) so
the interpretability methods see a uniform interface.

Aya Expanse is Cohere's CommandR/Cohere2 architecture: Llama-class decoder
with a custom attention scaling (1 / sqrt(d_head) is multiplied by an
extra factor inside the HF CohereAttention). For phase-one Q1 we don't
need to special-case it — the residual stream, MLP outputs, and W_O all
have the same shapes as Llama-2 — but Q2/Q3 (activation patching) should
sanity-check on a known induction head before trusting per-head numbers.

Tests: `tests/models/test_loaders.py::test_aya_*`.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ._hooked import LLAMA_PATHS, HookedModel


HF_NAME = "CohereForAI/aya-expanse-8b"

PAIR_TO_LANG_NAMES: dict[str, tuple[str, str]] = {
    "cs-de": ("Czech", "German"),
    "en-zh": ("English", "Chinese (Simplified)"),
    "en-arz": ("English", "Egyptian Arabic"),
}


def load_aya(*, dtype: str = "bfloat16", device: str = "cuda") -> HookedModel:
    """Load Aya Expanse 8B and wrap it for interp.

    Args:
        dtype: model dtype; bfloat16 recommended on A100.
        device: cuda or cpu.

    Returns:
        A `HookedModel` exposing the TL-shaped API used by src.interp.*.
    """
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(HF_NAME)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_NAME,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    hf_model.eval()
    # Cohere's HF config uses Llama-style attribute names (num_attention_heads,
    # hidden_size, num_hidden_layers) so LLAMA_PATHS works unchanged.
    return HookedModel(hf_model, tokenizer, LLAMA_PATHS)


def build_mt_prompt(source: str, pair: str) -> str:
    """Format an MT prompt for Aya (deterministic, single-turn).

    The prompt ends exactly where the model starts generating the target,
    so a logit lens read at the last token sees "what comes next" — which
    for Q1 is the target-language token mass we care about.
    """
    if pair not in PAIR_TO_LANG_NAMES:
        raise ValueError(f"Unknown pair {pair!r}; expected one of {list(PAIR_TO_LANG_NAMES)}.")
    src_lang, tgt_lang = PAIR_TO_LANG_NAMES[pair]
    return (
        f"Translate the following {src_lang} sentence into {tgt_lang}. "
        f"Output only the translation.\n\n"
        f"{src_lang}: {source}\n{tgt_lang}: "
    )


def tokenize_target_prefix(model: HookedModel, target: str, *, max_tokens: int = 8) -> list[int]:
    """First `max_tokens` ids of `target` (no special tokens)."""
    ids = model.tokenizer(target, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]


def build_mt_prompt(source: str, pair: str) -> str:
    """Format an MT prompt using Aya's chat template.

    Aya Expanse uses Cohere's chat-template format. For interpretability we
    want a deterministic single-turn prompt that ends right where the model
    is about to start generating the target. We render the chat template
    via the tokenizer's apply_chat_template when available, else fall back
    to a minimal Cohere-style format.

    Args:
        source: source sentence.
        pair: one of the keys in PAIR_TO_LANG_NAMES.
    """
    if pair not in PAIR_TO_LANG_NAMES:
        raise ValueError(f"Unknown pair {pair!r}; expected one of {list(PAIR_TO_LANG_NAMES)}.")
    src_lang, tgt_lang = PAIR_TO_LANG_NAMES[pair]
    return (
        f"Translate the following {src_lang} sentence into {tgt_lang}. "
        f"Output only the translation.\n\n"
        f"{src_lang}: {source}\n{tgt_lang}: "
    )


def tokenize_target_prefix(model: Any, target: str, *, max_tokens: int = 8) -> list[int]:
    """Return the first `max_tokens` token ids of `target` (no specials).

    Used by Q1 to track per-layer probability mass on the gold target prefix.
    Strips BOS/EOS so logit-lens mass tracks the actual continuation tokens.
    """
    tok = model.tokenizer
    ids = tok(target, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]
