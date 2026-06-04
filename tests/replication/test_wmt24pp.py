"""Pure-logic tests for the WMT24++ loader (no network/dataset access)."""

from __future__ import annotations

import pytest

from src.data.wmt24pp import (
    WMT24PP_LANGS,
    all_directions,
    parse_direction,
)


def test_parse_direction_both_ways():
    assert parse_direction("en-bn") == ("en", "bn")
    assert parse_direction("bn-en") == ("bn", "en")
    assert parse_direction("en-ja") == ("en", "ja")


def test_parse_direction_requires_english():
    with pytest.raises(ValueError):
        parse_direction("bn-fr")  # no English side


def test_parse_direction_unknown_language():
    with pytest.raises(ValueError):
        parse_direction("en-xx")


def test_all_directions_covers_six_langs_both_ways():
    dirs = all_directions()
    assert len(dirs) == 2 * len(WMT24PP_LANGS) == 12
    for lg in WMT24PP_LANGS:
        assert f"en-{lg}" in dirs
        assert f"{lg}-en" in dirs
