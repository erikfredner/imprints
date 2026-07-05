"""Tests for imprints.marc_place_geonames."""

from imprints import marc_place_geonames as mpg


def _admin1_fixture(tmp_path):
    """A tiny admin1CodesASCII.txt covering US/CA/GB rows used below."""
    path = tmp_path / "admin1CodesASCII.txt"
    path.write_text(
        "US.GA\tGeorgia\tGeorgia\t123\n"
        "US.NY\tNew York\tNew York\t124\n"
        "CA.10\tQuebec\tQuebec\t125\n"
        "GB.ENG\tEngland\tEngland\t126\n"
    )
    return str(path)


def test_strip_qualifier_only_strips_state_and_province():
    assert mpg._strip_qualifier("New York (State)") == "New York"
    assert mpg._strip_qualifier("Washington (State)") == "Washington"
    assert mpg._strip_qualifier("Quebec (Province)") == "Quebec"
    # "(Republic)" must NOT be stripped -- doing so would erase the
    # distinction MARC uses it for (see module docstring).
    assert mpg._strip_qualifier("Georgia (Republic)") == "Georgia (Republic)"


def test_normalize_for_match_strips_accents_and_lowercases():
    assert mpg._normalize_for_match("Québec") == "quebec"
    assert mpg._normalize_for_match("  New   York ") == "new york"


def test_resolve_scope_by_name_country(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA", "GB"])
    assert mpg.resolve_scope_by_name(
        "United States", admin1_names, ["US", "CA", "GB"]
    ) == (
        "US",
        None,
    )
    # Case-insensitive, as needed for imprints.llm_geocode's lowercase output.
    assert mpg.resolve_scope_by_name(
        "united states", admin1_names, ["US", "CA", "GB"]
    ) == (
        "US",
        None,
    )


def test_resolve_scope_by_name_admin1(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA", "GB"])
    assert mpg.resolve_scope_by_name("Georgia", admin1_names, ["US", "CA", "GB"]) == (
        "US",
        "GA",
    )
    assert mpg.resolve_scope_by_name("Quebec", admin1_names, ["US", "CA", "GB"]) == (
        "CA",
        "10",
    )
    # Accented MARC form still resolves via asciiname comparison.
    assert mpg.resolve_scope_by_name("Québec", admin1_names, ["US", "CA", "GB"]) == (
        "CA",
        "10",
    )


def test_resolve_scope_by_name_unresolved_returns_none(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA", "GB"])
    assert (
        mpg.resolve_scope_by_name("Atlantis", admin1_names, ["US", "CA", "GB"]) is None
    )


def test_build_crosswalk_resolves_and_freezes(tmp_path):
    admin1_path = _admin1_fixture(tmp_path)
    rows = mpg.build_crosswalk(
        ["New York (State)", "Georgia", "United States", "Atlantis"],
        admin1_path,
        ["US", "CA", "GB"],
    )
    by_name = {r["place_name_008"]: r for r in rows}
    assert by_name["New York (State)"] == {
        "place_name_008": "New York (State)",
        "geonames_country_code": "US",
        "geonames_admin1_code": "NY",
    }
    assert by_name["Georgia"]["geonames_admin1_code"] == "GA"
    assert by_name["United States"] == {
        "place_name_008": "United States",
        "geonames_country_code": "US",
        "geonames_admin1_code": "",
    }
    # Unresolved values are left blank, not fabricated.
    assert by_name["Atlantis"]["geonames_country_code"] == ""


def test_resolve_geo_scope_reads_frozen_csv(tmp_path):
    crosswalk_path = tmp_path / "crosswalk.csv"
    crosswalk_path.write_text(
        "place_name_008,geonames_country_code,geonames_admin1_code\n"
        "Georgia,US,GA\n"
        "United States,US,\n"
        "Atlantis,,\n"
    )
    mpg._load_frozen_crosswalk.cache_clear()
    assert mpg.resolve_geo_scope("Georgia", str(crosswalk_path)) == ("US", "GA")
    assert mpg.resolve_geo_scope("United States", str(crosswalk_path)) == ("US", None)
    assert mpg.resolve_geo_scope("Atlantis", str(crosswalk_path)) is None
    assert mpg.resolve_geo_scope(None, str(crosswalk_path)) is None
    assert mpg.resolve_geo_scope(float("nan"), str(crosswalk_path)) is None
    mpg._load_frozen_crosswalk.cache_clear()
