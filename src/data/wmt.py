"""WMT-style MT calibration data loader.

Phase-one calibration sets are small (≤2K source-target pairs per language
pair) — IFR's prior result averaged over 200 examples; activation patching
typically uses 50–200 paired prompts. This loader reads JSONL files
committed under the gitignored `data/` directory:

    data/cs-de.jsonl
    data/en-zh.jsonl
    data/en-arz.jsonl

Each line is `{"source": "...", "target": "...", "source_id": "..."}`.
The JSONL format is intentional: small calibration sets are easier to
ship as files than to wrangle HuggingFace dataset configs, and the WMT25
arz pair is non-standard enough that no off-the-shelf HF dataset covers
it cleanly.

`load_wmt_pairs(pair, n=...)` returns an iterable of `MTRecord`s. For
tests, `load_wmt_pairs_from_path(path, pair, n=...)` reads any file —
useful for tiny fixtures.

Tests: `tests/data/test_wmt.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Literal

LanguagePair = Literal["cs-de", "en-zh", "en-arz"]
LANGUAGE_PAIRS: tuple[LanguagePair, ...] = ("cs-de", "en-zh", "en-arz")

# Default location: the gitignored `data/` directory at the repo root.
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class MTRecord:
    """One source-target pair."""

    source: str
    target: str
    pair: LanguagePair
    source_id: str | None = None  # provenance, e.g., "wmt22.cs-de.123"


def load_wmt_pairs(
    pair: LanguagePair,
    *,
    n: int | None = None,
    data_dir: Path | None = None,
) -> Iterable[MTRecord]:
    """Iterate MT records for a language pair from `data/{pair}.jsonl`.

    Args:
        pair: one of "cs-de" | "en-zh" | "en-arz".
        n: maximum records to yield; default = all.
        data_dir: override the default `data/` directory (used by tests).
    """
    if pair not in LANGUAGE_PAIRS:
        raise ValueError(f"Unknown pair {pair!r}; expected one of {LANGUAGE_PAIRS}.")
    base = data_dir if data_dir is not None else DEFAULT_DATA_DIR
    path = base / f"{pair}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing calibration data: {path}. Populate it with one JSON object "
            f"per line: {{\"source\": ..., \"target\": ..., \"source_id\": ...}}."
        )
    return load_wmt_pairs_from_path(path, pair, n=n)


def load_wmt_pairs_from_path(
    path: Path,
    pair: LanguagePair,
    *,
    n: int | None = None,
) -> Iterator[MTRecord]:
    """Read MTRecords from an arbitrary JSONL path.

    Used directly by tests; production code calls `load_wmt_pairs`.
    """
    with path.open() as f:
        for i, line in enumerate(f):
            if n is not None and i >= n:
                return
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield MTRecord(
                source=data["source"],
                target=data["target"],
                pair=pair,
                source_id=data.get("source_id"),
            )
