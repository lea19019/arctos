"""AutoRound (Intel) quantization — paper's "AutoRound" method.

The paper configures AutoRound with 512 calibration samples, max sequence
length 4096, 512 optimization iterations, and group size 128 (4-bit) / 32
(2-bit). Supports both 4-bit and 2-bit. Quantizes once to disk; reloaded
through Transformers for shared ``hf_generate`` inference.

Calibration: AutoRound's default dataset (pile-10k) is fetched online and is
unavailable on offline compute nodes, so we pass an explicit ``calib_texts``
list (generic WikiText) — same corpus as the AWQ/GGUF calibration arms.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

SUPPORTED_BITS = (4, 2)

# Paper hyperparameters (overridable via env for fast smoke tests).
_ITERS = int(os.environ.get("AUTOROUND_ITERS", "512"))
_SEQLEN = int(os.environ.get("AUTOROUND_SEQLEN", "4096"))
_NSAMPLES = int(os.environ.get("AUTOROUND_NSAMPLES", "512"))
_GROUP_SIZE = {4: 128, 2: 32}


def _packed_chunks(texts: list[str], tokenizer, seqlen: int, n_samples: int) -> list[str]:
    """Concatenate texts and split into chunks of exactly ``seqlen`` tokens.

    AutoRound drops calibration samples shorter than ``seqlen``, so we pack the
    (short) source lines into full-length chunks. If the supplied pool yields
    fewer than ``n_samples`` chunks, top up from WikiText so calibration always
    has data and approximates the paper's sample count.
    """
    from .calib import wikitext_lines

    # Pack chunks slightly LONGER than seqlen: decoding ids->text then letting
    # AutoRound re-tokenize is not length-preserving (e.g. the Llama tokenizer
    # shrinks a 4096-id chunk back to ~4040 tokens), and AutoRound DROPS any
    # sample shorter than seqlen. A margin guarantees the re-tokenized text is
    # still >= seqlen, so chunks survive for every tokenizer (not just Qwen).
    chunk_len = seqlen + max(256, seqlen // 8)

    def chunkify(pool: list[str]) -> list[str]:
        ids: list[int] = []
        out: list[str] = []
        for t in pool:
            ids.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
            while len(ids) >= chunk_len:
                out.append(tokenizer.decode(ids[:chunk_len]))
                ids = ids[chunk_len:]
                if len(out) >= n_samples:
                    return out
        return out

    chunks = chunkify(texts)
    if len(chunks) < n_samples:
        # Need ~n_samples*seqlen tokens; pull a generous WikiText pool to fill.
        need_lines = max(2000, n_samples * max(1, seqlen // 24))
        chunks = chunkify(list(wikitext_lines(need_lines)))
    return chunks[:n_samples]


def quantize_to_disk(
    base_dir: str,
    bits: int,
    artifact_dir: str,
    calib_texts: Sequence[str],
) -> str:
    """Quantize ``base_dir`` with AutoRound at ``bits`` and save to disk."""
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"AutoRound supports {SUPPORTED_BITS}, got {bits}-bit.")
    import torch
    from auto_round import AutoRound
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(artifact_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, torch_dtype=torch.bfloat16, device_map="auto"  # spread 32B/70B
    )
    # AutoRound's `dataset` arg expects dataset *names* or a local json/jsonl
    # path (registered "local" loader) — NOT a list of raw strings (that parses
    # as dataset names, matches nothing, caches no data). Moreover AutoRound
    # DROPS any sample shorter than `seqlen`, and raw WikiText lines are far
    # shorter than 512/4096 tokens. So we PACK the calibration text into chunks
    # of exactly `seqlen` tokens (the standard concatenate-and-chunk recipe),
    # topping up from WikiText if the passed pool is too small, and write a
    # jsonl with a "text" field.
    calib_path = os.path.join(artifact_dir, "_ar_calib.jsonl")
    chunks = _packed_chunks(list(calib_texts), tokenizer, _SEQLEN, _NSAMPLES)
    with open(calib_path, "w", encoding="utf-8") as f:
        for text in chunks:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    ar = AutoRound(
        model,
        tokenizer,
        bits=bits,
        group_size=_GROUP_SIZE[bits],
        sym=True,
        iters=_ITERS,
        seqlen=_SEQLEN,
        nsamples=_NSAMPLES,
        dataset=calib_path,
    )
    ar.quantize()
    # Native auto_round format supports low-bit (incl. 2-bit) cleanly.
    ar.save_quantized(artifact_dir, format="auto_round")
    tokenizer.save_pretrained(artifact_dir)
    return artifact_dir


def load_model(artifact_dir: str, bits: int):
    """Load an AutoRound-quantized model for inference. Returns (model, tok)."""
    import torch

    # Importing auto_round registers its Transformers quantization backend.
    import auto_round  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(artifact_dir)
    model = AutoModelForCausalLM.from_pretrained(
        artifact_dir, device_map="cuda", torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer
