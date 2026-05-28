"""Clean/corrupt paired-prompt generators for activation patching.

A useful corrupt prompt for MT must:

1. Share template / instruction tokens with the clean prompt (so the
   patched computation is well-defined position-by-position).
2. Differ in a way that *would* change the gold translation, otherwise
   the patching effect is washed out — paraphrasing the source rarely
   satisfies this for MT, since paraphrases often have identical
   translations.
3. Be in distribution — gibberish corrupt prompts conflate "MT circuit"
   with "general LM circuit" and the patching effect attributes both.

Per `PHASE1-PLAN.md` risk register: do not trust a single corrupt-prompt
design. Use multiple generators and require that activation-patching
effect sizes are stable across them before reporting per-head rankings.

Three strategies are implemented here as **minimum-viable, in-distribution
substitutions** — research-grade variants (POS-aware lexical sub, attested
cross-lingual neighbors, target-language preserving paraphrases) are the
user's responsibility. The strategies as implemented satisfy criteria
(1)–(3) for short calibration sentences but should be inspected before
trusting any per-head ranking they produce.

Tests: `tests/data/test_clean_corrupt.py`.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .wmt import MTRecord


class CorruptStrategy(str, Enum):
    """How to derive the corrupt source from the clean source."""

    LEXICAL_SUB = "lexical_sub"        # swap one content word for one drawn from another corpus record
    LANG_ID_SWAP = "lang_id_swap"      # replace one source token with a token from a different language
    TARGET_SHUFFLE = "target_shuffle"  # swap two content words in the source


@dataclass(frozen=True)
class CleanCorruptPair:
    """A paired prompt usable for activation patching.

    Both `clean` and `corrupt` carry an MT-format record (source + target).
    The corrupt's `target` is the *expected* gold target for the corrupted
    source — for these minimum-viable strategies we leave it equal to the
    clean target as a placeholder; activation-patching metrics that need
    distinct corrupt targets (LOGIT_DIFF in particular) should pass the
    expected corrupt-target token explicitly to `activation_patch`.
    """

    clean: MTRecord
    corrupt: MTRecord
    strategy: CorruptStrategy


# Conservative content-word filter — anything ≥3 chars and not a stopword.
# Phase-one calibration sentences are short; we do not need anything more
# sophisticated here. Real POS-aware filtering belongs in a research-grade
# successor, not this stub.
_STOPWORDS = frozenset(
    {
        # English
        "the", "and", "for", "with", "that", "this", "from", "have", "are",
        "was", "were", "but", "not", "will", "you", "all", "can", "had",
        # Czech
        "jsem", "jsi", "jsou", "byl", "byla", "byli", "ale", "nebo", "tak",
        "jako", "ktery", "která", "které", "že", "se", "po", "do", "za",
    }
)


def _word_tokens(text: str) -> list[tuple[int, int, str]]:
    """Yield (start, end, token) for each whitespace-delimited word.

    We deliberately avoid a real tokenizer here: the corrupt strategies
    operate at the source-text level, before any model-specific tokenizer
    runs. Punctuation is left attached to its word.
    """
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def _is_content_word(tok: str) -> bool:
    bare = re.sub(r"[^\w]", "", tok).lower()
    return len(bare) >= 3 and bare not in _STOPWORDS


def _bare_word(tok: str) -> str:
    """Strip outer (leading/trailing) non-word characters from a token.

    `_word_tokens` returns the raw whitespace-delimited token including any
    trailing punctuation, e.g., "sings." — for use as a *donor* word in a
    different sentence we want only the bare lexical content.
    """
    return re.sub(r"^[^\w]+|[^\w]+$", "", tok)


def _replace_token(text: str, span: tuple[int, int], replacement: str) -> str:
    start, end = span
    return text[:start] + replacement + text[end:]


class CleanCorruptGenerator:
    """Builds clean/corrupt prompt pairs given a corpus of in-distribution records.

    The corpus serves two purposes:
    - LEXICAL_SUB samples its replacement word from another record in the
      corpus (so the substituted word is in distribution for the language
      pair, not noise).
    - LANG_ID_SWAP samples a token from a record in a *different* language
      pair (so the inserted word has the right "shape" of a foreign token).

    `seed` makes a generator deterministic for tests and for reproducing
    a specific corrupt set.
    """

    def __init__(self, corpus: Sequence[MTRecord], *, seed: int = 0) -> None:
        self.corpus = list(corpus)
        self.rng = random.Random(seed)

    # --- public API ---------------------------------------------------------

    def make_pair(self, record: MTRecord, *, strategy: CorruptStrategy) -> CleanCorruptPair:
        """Construct a single clean/corrupt pair from one MT record."""
        if strategy is CorruptStrategy.LEXICAL_SUB:
            corrupt_source = self._lexical_sub(record)
        elif strategy is CorruptStrategy.LANG_ID_SWAP:
            corrupt_source = self._lang_id_swap(record)
        elif strategy is CorruptStrategy.TARGET_SHUFFLE:
            corrupt_source = self._target_shuffle(record)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        corrupt = MTRecord(
            source=corrupt_source,
            target=record.target,  # placeholder; see CleanCorruptPair docstring
            pair=record.pair,
            source_id=(record.source_id + ".corrupt") if record.source_id else None,
        )
        return CleanCorruptPair(clean=record, corrupt=corrupt, strategy=strategy)

    # --- strategies ---------------------------------------------------------

    def _content_word_spans(self, text: str) -> list[tuple[int, int, str]]:
        """All content-word spans in the source; raises if there are none."""
        spans = [s for s in _word_tokens(text) if _is_content_word(s[2])]
        if not spans:
            raise ValueError(
                f"No content words found in source {text!r}; cannot apply "
                f"corrupt strategy. Pre-filter the corpus or extend _STOPWORDS."
            )
        return spans

    def _lexical_sub(self, record: MTRecord) -> str:
        """Replace one content word with a content word drawn from another record in the same pair.

        In-distribution by construction: the replacement is a real word from
        the same WMT calibration corpus. Note: if the donor word is the same
        as the original (rare but possible), we resample up to 5 times.
        """
        clean_spans = self._content_word_spans(record.source)
        target_span = self.rng.choice(clean_spans)
        original = target_span[2]

        # Donors: bare content words from other records in the same language pair.
        donors = [
            _bare_word(tok)
            for r in self.corpus
            if r.pair == record.pair and r.source_id != record.source_id
            for _, _, tok in self._content_word_spans(r.source)
        ] or [_bare_word(original)]

        bare_original = _bare_word(original)
        replacement = bare_original
        for _ in range(5):
            replacement = self.rng.choice(donors)
            if replacement.lower() != bare_original.lower():
                break

        return _replace_token(record.source, (target_span[0], target_span[1]), replacement)

    def _lang_id_swap(self, record: MTRecord) -> str:
        """Replace a single source token with one drawn from a different language pair.

        Inserts cross-script signal: e.g., for a cs-de record, the donor is
        drawn from en-zh's source side (English) or en-arz's source side
        (English) or vice versa. This produces a corrupt prompt that *looks
        like* the wrong source language to the model.
        """
        clean_spans = self._content_word_spans(record.source)
        target_span = self.rng.choice(clean_spans)

        donors = [
            _bare_word(tok)
            for r in self.corpus
            if r.pair != record.pair
            for _, _, tok in self._content_word_spans(r.source)
        ]
        if not donors:
            # Fallback: a Chinese marker so even a same-pair-only corpus
            # still produces a cross-script perturbation that the test can
            # observe.
            donors = ["猫", "犬", "山"]
        replacement = self.rng.choice(donors)

        return _replace_token(record.source, (target_span[0], target_span[1]), replacement)

    def _target_shuffle(self, record: MTRecord) -> str:
        """Swap two content words in the source.

        Keeps source-language identity intact; changes word order, which
        for most language pairs changes the gold translation. (For the
        rare case where the source has only one content word, falls back
        to LEXICAL_SUB.)
        """
        clean_spans = self._content_word_spans(record.source)
        if len(clean_spans) < 2:
            return self._lexical_sub(record)

        i, j = self.rng.sample(range(len(clean_spans)), 2)
        # Apply the later substitution first so earlier indices stay valid.
        if i > j:
            i, j = j, i
        si, ei, ti = clean_spans[i]
        sj, ej, tj = clean_spans[j]
        new = (
            record.source[:si] + tj + record.source[ei:sj] + ti + record.source[ej:]
        )
        return new


def make_pair(
    record: MTRecord,
    *,
    strategy: CorruptStrategy,
    corpus: Sequence[MTRecord] | None = None,
    seed: int = 0,
) -> CleanCorruptPair:
    """Module-level convenience wrapper around `CleanCorruptGenerator`.

    For repeated use, prefer constructing a `CleanCorruptGenerator` directly
    so the corpus is loaded once.
    """
    return CleanCorruptGenerator(corpus or [record], seed=seed).make_pair(
        record, strategy=strategy
    )
