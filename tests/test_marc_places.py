"""Tests for imprints.marc_places."""

from imprints import marc_places as mp


def test_decode_marc_country_known_codes():
    assert mp.decode_marc_country("nyu") == "New York (State)"
    assert mp.decode_marc_country("enk") == "England"
    assert mp.decode_marc_country("xxr") == "Soviet Union"


def test_decode_marc_country_unknown_code_returns_none():
    assert mp.decode_marc_country("zz9") is None


def test_decode_marc_country_blank_or_none_returns_none():
    assert mp.decode_marc_country(None) is None
    assert mp.decode_marc_country("") is None
