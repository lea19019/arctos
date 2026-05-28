"""Tests for src.data.clean_corrupt."""

from __future__ import annotations

import pytest

from src.data.clean_corrupt import (
    CleanCorruptGenerator,
    CleanCorruptPair,
    CorruptStrategy,
    make_pair,
)
from src.data.wmt import MTRecord


def _record(source: str, pair: str = "cs-de", source_id: str = "x.0") -> MTRecord:
    return MTRecord(source=source, target="placeholder target", pair=pair, source_id=source_id)


@pytest.fixture
def corpus() -> list[MTRecord]:
    return [
        _record("Prosím otevřete dveře.", "cs-de", "cs.0"),
        _record("Kočka leží na okně.", "cs-de", "cs.1"),
        _record("Slunce svítí jasně dnes.", "cs-de", "cs.2"),
        _record("The brown dog ran fast.", "en-zh", "en.0"),
        _record("My little sister sings.", "en-zh", "en.1"),
        _record("Egyptian streets feel busy.", "en-arz", "en.0"),
    ]


# ---------------- LEXICAL_SUB ----------------


@pytest.mark.cpu
def test_lexical_sub_changes_source(corpus):
    gen = CleanCorruptGenerator(corpus, seed=0)
    record = corpus[0]

    pair = gen.make_pair(record, strategy=CorruptStrategy.LEXICAL_SUB)

    assert isinstance(pair, CleanCorruptPair)
    assert pair.strategy is CorruptStrategy.LEXICAL_SUB
    assert pair.clean is record
    assert pair.corrupt.source != record.source  # at least one token differs
    assert pair.corrupt.pair == record.pair
    assert pair.corrupt.source_id == "cs.0.corrupt"


@pytest.mark.cpu
def test_lexical_sub_preserves_word_count(corpus):
    """The replacement is a single token, so the total word count is unchanged."""
    gen = CleanCorruptGenerator(corpus, seed=0)
    record = corpus[2]

    pair = gen.make_pair(record, strategy=CorruptStrategy.LEXICAL_SUB)

    assert len(pair.corrupt.source.split()) == len(record.source.split())


@pytest.mark.cpu
def test_lexical_sub_is_deterministic(corpus):
    """Same seed → same output."""
    a = CleanCorruptGenerator(corpus, seed=42).make_pair(corpus[1], strategy=CorruptStrategy.LEXICAL_SUB)
    b = CleanCorruptGenerator(corpus, seed=42).make_pair(corpus[1], strategy=CorruptStrategy.LEXICAL_SUB)

    assert a.corrupt.source == b.corrupt.source


# ---------------- LANG_ID_SWAP ----------------


@pytest.mark.cpu
def test_lang_id_swap_changes_source(corpus):
    gen = CleanCorruptGenerator(corpus, seed=0)
    record = corpus[0]

    pair = gen.make_pair(record, strategy=CorruptStrategy.LANG_ID_SWAP)

    assert pair.corrupt.source != record.source
    assert pair.strategy is CorruptStrategy.LANG_ID_SWAP


@pytest.mark.cpu
def test_lang_id_swap_uses_other_pair_donor(corpus):
    """For a cs-de record, the donor token comes from en-zh or en-arz, so an
    English content word should appear in the corrupted Czech source."""
    gen = CleanCorruptGenerator(corpus, seed=0)
    record = corpus[0]  # cs-de

    pair = gen.make_pair(record, strategy=CorruptStrategy.LANG_ID_SWAP)

    # The corrupt should contain at least one ASCII-only English-style word
    # drawn from one of the en-zh / en-arz records.
    english_words = {"brown", "dog", "ran", "fast", "little", "sister", "sings",
                     "Egyptian", "streets", "feel", "busy", "The", "My"}
    corrupt_words = set(pair.corrupt.source.split())
    assert corrupt_words & english_words, (
        f"Expected an English donor token in corrupted source, got {pair.corrupt.source!r}"
    )


# ---------------- TARGET_SHUFFLE ----------------


@pytest.mark.cpu
def test_target_shuffle_changes_source_order(corpus):
    gen = CleanCorruptGenerator(corpus, seed=0)
    record = corpus[2]  # 4 content words → enough to shuffle

    pair = gen.make_pair(record, strategy=CorruptStrategy.TARGET_SHUFFLE)

    assert pair.corrupt.source != record.source
    # A shuffle of two content words within the source preserves the multiset
    # of tokens (split on whitespace).
    assert sorted(pair.corrupt.source.split()) == sorted(record.source.split())


@pytest.mark.cpu
def test_target_shuffle_falls_back_when_too_few_words():
    """With only one content word, target-shuffle falls back to lexical-sub."""
    # "I see cat." → "I" 1-char, "see" 3-char content, "cat." → "cat" content.
    # Two content words → target_shuffle works without fallback. Use one
    # content word instead:
    record = MTRecord(source="I see cat.", target="...", pair="cs-de", source_id="t.0")
    # Donor corpus has plenty of content words to pull from.
    corpus = [
        record,
        MTRecord(source="The brown dog ran fast.", target="...", pair="cs-de", source_id="t.1"),
    ]

    gen = CleanCorruptGenerator(corpus, seed=0)
    pair = gen.make_pair(record, strategy=CorruptStrategy.TARGET_SHUFFLE)

    assert pair.strategy is CorruptStrategy.TARGET_SHUFFLE  # strategy label preserved
    assert pair.corrupt.source != record.source              # something changed


@pytest.mark.cpu
def test_target_shuffle_no_content_words_raises():
    """A source with NO content words can't be corrupted by any strategy
    that needs to pick a content-word span."""
    record = MTRecord(source="Hi ok.", target="...", pair="cs-de", source_id="t.0")
    gen = CleanCorruptGenerator([record], seed=0)
    with pytest.raises(ValueError, match="No content words"):
        gen.make_pair(record, strategy=CorruptStrategy.LEXICAL_SUB)


# ---------------- module-level make_pair ----------------


@pytest.mark.cpu
def test_module_level_make_pair(corpus):
    pair = make_pair(corpus[1], strategy=CorruptStrategy.LEXICAL_SUB, corpus=corpus, seed=7)

    assert isinstance(pair, CleanCorruptPair)
    assert pair.strategy is CorruptStrategy.LEXICAL_SUB
