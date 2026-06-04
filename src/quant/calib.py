"""Calibration text for the quantizers.

Two calibration regimes, matching the paper:

  * **Generic** (the default for AWQ, AutoRound, and the main-sweep GGUF
    imatrix): WikiText, as the paper uses for its imatrix estimation. Used for
    every method's "generic" arm.
  * **Language-matched** (the C3 deep-dive): monolingual text in the target
    language from FineWeb-2, exactly as the paper builds its English-vs-Bengali
    calibration sets.

All datasets must be pre-cached on the login node (see ``precache.py``); these
loaders run offline on compute nodes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

# Paper: imatrix from 20k WikiText samples; AWQ/AutoRound use far fewer.
WIKITEXT_NAME = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"

# FineWeb-2 per-language configs (script-tagged), for the C3 calibration arm.
# NOTE / red flag: the paper states it sampled *English* calibration text from
# "HuggingFaceFW/fineweb-2", but FineWeb-2 has NO English config (it is the
# explicitly multilingual sibling of FineWeb; there is no `eng_Latn`). English
# web text lives in the original FineWeb. We therefore source English from
# FineWeb and the non-English languages from FineWeb-2, and flag the paper's
# (impossible-as-stated) English source in the findings doc.
FINEWEB2_NAME = "HuggingFaceFW/fineweb-2"
FINEWEB2_CONFIG = {
    "bn": "ben_Beng",
    "fr": "fra_Latn",
    "ml": "mal_Mlym",
}
FINEWEB_EN = ("HuggingFaceFW/fineweb", "sample-10BT")  # English: original FineWeb


@lru_cache(maxsize=2)
def wikitext_lines(n: int) -> tuple[str, ...]:
    """Return ``n`` non-trivial WikiText lines (deduped, length-filtered)."""
    from datasets import load_dataset

    ds = load_dataset(WIKITEXT_NAME, WIKITEXT_CONFIG, split="train")
    out: list[str] = []
    seen: set[str] = set()
    for row in ds:
        line = row["text"].strip()
        # Skip blank lines and section headers like " = = Title = = ".
        if len(line) < 32 or line.startswith("="):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
        if len(out) >= n:
            break
    return tuple(out)


def fineweb2_text(lang: str, n_tokens: int, tokenizer) -> str:
    """Concatenated FineWeb-2 text in ``lang`` totalling ~``n_tokens`` tokens.

    ``tokenizer`` is the model tokenizer, used only to count tokens so the
    English and language-matched calibration sets are the same *token* size
    (the paper samples "10k tokens" per language).
    """
    from datasets import load_dataset

    if lang == "en":
        ds = load_dataset(FINEWEB_EN[0], FINEWEB_EN[1], split="train", streaming=True)
    elif lang in FINEWEB2_CONFIG:
        ds = load_dataset(
            FINEWEB2_NAME, FINEWEB2_CONFIG[lang], split="train", streaming=True
        )
    else:
        raise ValueError(
            f"No FineWeb config for {lang!r}; have en + {list(FINEWEB2_CONFIG)}.")
    chunks: list[str] = []
    total = 0
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue
        chunks.append(text)
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if total >= n_tokens:
            break
    return "\n".join(chunks)


def write_imatrix_file(lines, path: str) -> str:
    """Write calibration lines to a text file for ``llama-imatrix -f``."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
