"""NLLB-200 loader (Meta, encoder-decoder MT — the architectural outlier).

NLLB is a BART-style **encoder-decoder**, not a decoder-only causal LM, so
it does not fit the `HookedModel` interface the other 7 models share. But
the core Q1 question — "where in the decoder does target commitment happen"
— transfers cleanly, because the decoder is still an additive residual
stack and HF exposes `decoder_hidden_states` directly.

This loader returns the raw seq2seq model + tokenizer + the FLORES-style
language-code map NLLB expects (no instruction prompt; the source language
is set on the tokenizer and the target language is the forced first decoder
token).

Why NLLB matters for the quantization story: it's the only MT-purpose-built
*encoder-decoder* in the set. If the method's findings (depth signature,
importance/sensitivity relationship) differ for enc-dec vs decoder-only,
that's a boundary the paper must state.

Cite: NLLB Team (2022), "No Language Left Behind", arXiv:2207.04672.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

HF_NAME = "facebook/nllb-200-3.3B"

# FLORES-200 language codes NLLB uses, per project pair.
PAIR_TO_NLLB_CODES: dict[str, tuple[str, str]] = {
    "cs-de": ("ces_Latn", "deu_Latn"),
    "en-zh": ("eng_Latn", "zho_Hans"),
    "en-arz": ("eng_Latn", "arz_Arab"),
}


def load_nllb(*, dtype: str = "bfloat16", device: str = "cuda", hf_name: str = HF_NAME):
    """Return (hf_seq2seq_model, tokenizer). Not a HookedModel — enc-dec."""
    torch_dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(hf_name, torch_dtype=torch_dtype, device_map=device)
    model.eval()
    return model, tokenizer
