"""Shared greedy chat-template generation for the HF quantization paths.

AWQ, bitsandbytes, and AutoRound all produce models that run through the
standard HF ``generate`` API. This module is the single batched, greedy
(temperature=0, no sampling) decode loop they share, so the only thing that
differs between methods is *how the weights were quantized* — not decoding.

Prompts are rendered by ``src.models._chat_prompt.render_chat_prompt`` (the
same renderer the GGUF path uses) to keep the two backends comparable.
"""

from __future__ import annotations

from typing import Sequence

import torch

from src.data.wmt24pp import TranslationExample
from src.models._chat_prompt import render_chat_prompt

# Generous cap: WMT24++ segments are sentence-ish; 512 new tokens covers even
# verbose scripts without runaway generation. EOS stops earlier in practice.
DEFAULT_MAX_NEW_TOKENS = 512


def translate(
    model,
    tokenizer,
    examples: Sequence[TranslationExample],
    *,
    chat_kwargs: dict | None = None,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_size: int = 16,
) -> list[str]:
    """Greedy-decode translations for a list of examples (single direction).

    Returns one hypothesis string per example, in input order.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only: left-pad for batched gen
    device = next(model.parameters()).device

    hyps: list[str] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        prompts = [
            render_chat_prompt(
                tokenizer, ex.source, ex.src_lang, ex.tgt_lang, chat_kwargs=chat_kwargs
            )
            for ex in chunk
        ]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,  # template already added specials
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                do_sample=False,
                num_beams=1,
                temperature=None,
                top_p=None,
                top_k=None,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = out[:, enc["input_ids"].shape[1] :]
        for row in gen:
            text = tokenizer.decode(row, skip_special_tokens=True)
            hyps.append(_clean(text))
    return hyps


def _clean(text: str) -> str:
    """Trim whitespace and a leading 'Translation:'-style preamble if present."""
    text = text.strip()
    # Some chat models echo a short lead-in despite the instruction; strip the
    # most common one conservatively (only when it's the very first token run).
    for lead in ("Translation:", "translation:", "Sure, here", "Here is"):
        if text.startswith(lead):
            nl = text.find("\n")
            text = text[nl + 1 :].strip() if nl != -1 else text[len(lead) :].strip()
            break
    return text
