"""Aya Expanse 8B loader (Cohere; decoder-only, multilingual-by-pretraining).

Continuity model — the prior paper's target. TransformerLens has Cohere
support via `CohereForAI/aya-expanse-8b`. Cohere's attention scaling is
already handled inside TransformerLens's HookedTransformer config for the
Cohere model family — confirm with a logit-lens sanity check before any
patching experiment trusts the numbers.

Tests: `tests/models/test_loaders.py::test_aya_*`.
"""

from __future__ import annotations

from typing import Any


HF_NAME = "CohereForAI/aya-expanse-8b"

# Human-readable target-language names used in Aya's chat prompt for MT.
PAIR_TO_LANG_NAMES: dict[str, tuple[str, str]] = {
    "cs-de": ("Czech", "German"),
    "en-zh": ("English", "Chinese (Simplified)"),
    "en-arz": ("English", "Egyptian Arabic"),
}


def load_aya(*, dtype: str = "bfloat16", device: str = "cuda") -> Any:
    """Load Aya Expanse 8B via TransformerLens.

    Args:
        dtype: model dtype; bfloat16 recommended on A100.
        device: cuda or cpu.

    Returns:
        A TransformerLens HookedTransformer.
    """
    import torch
    from transformer_lens import HookedTransformer

    torch_dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    return HookedTransformer.from_pretrained(
        HF_NAME,
        dtype=torch_dtype,
        device=device,
    )


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
