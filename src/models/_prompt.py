"""Shared MT prompt builder used by all model loaders.

Each model has its own preferred chat template (Aya's Cohere-style,
Tower-Plus's Gemma2-style, Llama-3.1's `<|begin_of_text|>` header,
etc.), but for phase-one comparisons we use a single generic plain-text
prompt across all decoder-only models so results are
apples-to-apples — the *method* is what we're studying, not each
model's optimal MT prompt.

For the quantization story, generic-prompt results give a conservative
baseline: a model-tuned chat template would lift translation quality
modestly but shouldn't change which components are MT-critical.
"""

from __future__ import annotations

PAIR_TO_LANG_NAMES: dict[str, tuple[str, str]] = {
    "cs-de": ("Czech", "German"),
    "en-zh": ("English", "Chinese (Simplified)"),
    "en-arz": ("English", "Egyptian Arabic"),
}


def build_mt_prompt(source: str, pair: str) -> str:
    """Plain-text MT instruction prompt ending where the model starts generating."""
    if pair not in PAIR_TO_LANG_NAMES:
        raise ValueError(f"Unknown pair {pair!r}; expected one of {list(PAIR_TO_LANG_NAMES)}.")
    src_lang, tgt_lang = PAIR_TO_LANG_NAMES[pair]
    return (
        f"Translate the following {src_lang} sentence into {tgt_lang}. "
        f"Output only the translation.\n\n"
        f"{src_lang}: {source}\n{tgt_lang}: "
    )


def tokenize_target_prefix(model, target: str, *, max_tokens: int = 8) -> list[int]:
    """First `max_tokens` ids of `target` (no special tokens). Works with any HookedModel."""
    ids = model.tokenizer(target, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]
