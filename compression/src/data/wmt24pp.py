"""WMT24++ test-set loader for the PTQ-MT replication (arXiv:2508.20893).

This is a *replication* effort, kept deliberately separate from the Arctos
q6 work (see docs/replication-uneven-ptq-mt-brief.md). The paper evaluates on
WMT24++ (`google/wmt24pp`, Deutsch et al. 2025): 55 language pairs, all
sourced *from English*, across four domains (literary, news, social, speech).

Dataset schema (one HF config per target language, named ``en-XX_YY``):

    lp, domain, document_id, segment_id, is_bad_source, source, target,
    original_target

``source`` is the English side; ``target`` is the human post-edited
reference in the target language; ``original_target`` is the pre-post-edit
MT output (unused). Sentinel "canary" rows and rows flagged
``is_bad_source`` are dropped.

The paper translates in *both* directions to/from English. We model a
direction as a short code like ``en-ja`` or ``ja-en``:

* ``en-XX``: input = English ``source``, reference = ``target``.
* ``XX-en``: input = ``target`` (target-language text), reference = ``source``.
  As the paper notes, Xx->En is an artificial setting (the source is
  post-edited translationese), but they evaluate it, so we do too.

Offline use (SLURM compute nodes): the config must already be in the HF
datasets cache (pull it on the login node via ``precache.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

# Short language code -> WMT24++ target-config language code. The six
# "representative" languages the paper reports (Table 1-4), chosen for script
# diversity and varying training representation.
WMT24PP_LANGS: dict[str, str] = {
    "ja": "ja_JP",  # Japanese  (high-resource, Kanji/Kana)
    "fr": "fr_FR",  # French    (high-resource, Latin)
    "pl": "pl_PL",  # Polish    (mid-resource, Latin)
    "bn": "bn_IN",  # Bengali   (low-resource, Bengali script)
    "ml": "ml_IN",  # Malayalam (low-resource, Malayalam script)
    "zu": "zu_ZA",  # Zulu      (low-resource, Latin)
}

# Canary sentinel rows carry this domain (and GUID payloads); always skip them.
_CANARY_DOMAIN = "canary"


@dataclass(frozen=True)
class TranslationExample:
    """One translation instance: input text + reference, with provenance."""

    source: str          # the text to translate (input to the model)
    reference: str        # gold reference translation
    direction: str        # e.g. "en-ja" or "ja-en"
    src_lang: str         # short code of the input language, e.g. "en"
    tgt_lang: str         # short code of the output language, e.g. "ja"
    domain: str
    segment_id: int


def parse_direction(direction: str) -> tuple[str, str]:
    """Split ``"en-ja"`` -> ``("en", "ja")`` and validate against WMT24++.

    Exactly one side must be English; the other must be a known target lang.
    """
    try:
        src, tgt = direction.split("-")
    except ValueError as exc:  # pragma: no cover - guard
        raise ValueError(f"Bad direction {direction!r}; expected 'src-tgt'.") from exc
    if "en" not in (src, tgt):
        raise ValueError(f"Direction {direction!r} must include English (en).")
    other = tgt if src == "en" else src
    if other not in WMT24PP_LANGS:
        raise ValueError(
            f"Unknown language {other!r} in {direction!r}; "
            f"known: {sorted(WMT24PP_LANGS)}."
        )
    return src, tgt


def _config_for(direction: str) -> str:
    """The single HF config that backs a direction (always ``en-XX_YY``)."""
    src, tgt = parse_direction(direction)
    other = tgt if src == "en" else src
    return f"en-{WMT24PP_LANGS[other]}"


def load_wmt24pp(
    direction: str,
    *,
    n: int | None = None,
    domains: tuple[str, ...] | None = None,
) -> list[TranslationExample]:
    """Load up to ``n`` clean examples for a translation direction.

    Args:
        direction: short code such as "en-bn" or "bn-en".
        n: cap on returned examples (post-filtering); ``None`` = all.
        domains: optional whitelist of WMT24++ domains (literary/news/
            social/speech); ``None`` keeps all non-canary domains.

    Returns a list (not a generator): n<=~1000/direction is small, and the
    caller wants a stable, indexable, length-known sequence for scoring.
    """
    from datasets import load_dataset

    src, tgt = parse_direction(direction)
    config = _config_for(direction)
    ds = load_dataset("google/wmt24pp", config, split="train")

    out: list[TranslationExample] = []
    for row in ds:
        if row["domain"] == _CANARY_DOMAIN or row["is_bad_source"]:
            continue
        if domains is not None and row["domain"] not in domains:
            continue
        if src == "en":
            text, ref = row["source"], row["target"]
        else:
            text, ref = row["target"], row["source"]
        if not text or not ref:
            continue
        out.append(
            TranslationExample(
                source=text,
                reference=ref,
                direction=direction,
                src_lang=src,
                tgt_lang=tgt,
                domain=row["domain"],
                segment_id=int(row["segment_id"]),
            )
        )
        if n is not None and len(out) >= n:
            break
    return out


def all_directions(langs: tuple[str, ...] | None = None) -> list[str]:
    """Every en<->X direction for the given (default: all six) languages."""
    langs = langs or tuple(WMT24PP_LANGS)
    dirs: list[str] = []
    for lg in langs:
        dirs.append(f"en-{lg}")
        dirs.append(f"{lg}-en")
    return dirs
