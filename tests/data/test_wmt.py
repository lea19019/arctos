"""Tests for src.data.wmt.

CPU invariants only — no GPU needed for data loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.wmt import (
    LANGUAGE_PAIRS,
    MTRecord,
    load_wmt_pairs,
    load_wmt_pairs_from_path,
)


def _write_jsonl(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


@pytest.mark.cpu
def test_load_from_path_yields_records(tmp_path: Path):
    rows = [
        {"source": "Hello world.", "target": "Hallo Welt.", "source_id": "ex.0"},
        {"source": "Goodbye.", "target": "Auf Wiedersehen."},  # no source_id
    ]
    path = _write_jsonl(tmp_path, "cs-de.jsonl", rows)

    records = list(load_wmt_pairs_from_path(path, "cs-de"))

    assert len(records) == 2
    assert all(isinstance(r, MTRecord) for r in records)
    assert records[0].source == "Hello world."
    assert records[0].target == "Hallo Welt."
    assert records[0].pair == "cs-de"
    assert records[0].source_id == "ex.0"
    assert records[1].source_id is None  # missing key → None


@pytest.mark.cpu
def test_load_respects_n_limit(tmp_path: Path):
    rows = [{"source": f"src {i}", "target": f"tgt {i}"} for i in range(5)]
    path = _write_jsonl(tmp_path, "en-zh.jsonl", rows)

    records = list(load_wmt_pairs_from_path(path, "en-zh", n=3))

    assert len(records) == 3
    assert [r.source for r in records] == ["src 0", "src 1", "src 2"]


@pytest.mark.cpu
def test_load_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "en-arz.jsonl"
    p.write_text(
        '{"source": "a", "target": "b"}\n'
        "\n"
        '{"source": "c", "target": "d"}\n'
        "   \n"
    )

    records = list(load_wmt_pairs_from_path(p, "en-arz"))

    assert [r.source for r in records] == ["a", "c"]


@pytest.mark.cpu
def test_load_wmt_pairs_unknown_pair_raises():
    with pytest.raises(ValueError, match="Unknown pair"):
        list(load_wmt_pairs("xx-yy"))  # type: ignore[arg-type]


@pytest.mark.cpu
def test_load_wmt_pairs_uses_data_dir_override(tmp_path: Path):
    rows = [{"source": "a", "target": "b"}]
    _write_jsonl(tmp_path, "cs-de.jsonl", rows)

    records = list(load_wmt_pairs("cs-de", data_dir=tmp_path))

    assert len(records) == 1
    assert records[0].source == "a"


@pytest.mark.cpu
def test_load_wmt_pairs_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Missing calibration data"):
        list(load_wmt_pairs("cs-de", data_dir=tmp_path))


@pytest.mark.cpu
def test_language_pairs_constant_is_complete():
    """Sanity: the LANGUAGE_PAIRS tuple lists exactly the three phase-one pairs."""
    assert set(LANGUAGE_PAIRS) == {"cs-de", "en-zh", "en-arz"}
