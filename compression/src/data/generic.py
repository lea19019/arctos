"""Generic (non-MT) calibration text — the control for MT-conditional calibration.

The phase-two linchpin question (docs/research.md §1D): does calibrating a
quantizer on MT data pick better weights to protect than calibrating on generic
text? To answer it we need a *generic* calibration corpus in the *same
languages* the model sees during MT, so the contrast isolates task/domain — not
language.

We use XNLI premises (cached offline): generic NLI sentences spanning fiction,
travel, government, and telephone domains — emphatically not translation data.
Languages available offline: en, de, zh, ar (we map arz->ar as the closest
script/lang for the en-arz pair). This mirrors the MT calibration languages:
cs-de -> {de,en}, en-zh -> {en,zh}, en-arz -> {en,ar}.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Languages we can source generically, keyed to the MT pair languages we care
# about. cs is not in XNLI; de/en cover the cs-de target side + English pivot.
PAIR_TO_GENERIC_LANGS = {
    "cs-de": ("de", "en"),
    "en-zh": ("en", "zh"),
    "en-arz": ("en", "ar"),
}


@lru_cache(maxsize=8)
def _xnli_premises(lang: str, split: str = "validation") -> tuple[str, ...]:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset
    ds = load_dataset("xnli", lang, split=split)
    # dedupe premises (XNLI repeats each premise across 3 hypotheses)
    seen, out = set(), []
    for p in ds["premise"]:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def load_generic_calib(n: int, langs: tuple[str, ...]) -> list[str]:
    """Return ~n generic sentences, round-robin across `langs` (deduped)."""
    pools = {lg: _xnli_premises(lg) for lg in langs}
    out: list[str] = []
    i = 0
    while len(out) < n and any(i < len(pools[lg]) for lg in langs):
        for lg in langs:
            if i < len(pools[lg]):
                out.append(pools[lg][i])
                if len(out) >= n:
                    break
        i += 1
    return out


def generic_calib_for_pairs(pairs: list[str], n: int) -> list[str]:
    """Generic calibration matched to the languages of the given MT pairs."""
    langs: list[str] = []
    for p in pairs:
        for lg in PAIR_TO_GENERIC_LANGS.get(p, ("en",)):
            if lg not in langs:
                langs.append(lg)
    return load_generic_calib(n, tuple(langs))
