"""Tests for imprints.geonames_geocode."""

from imprints import geonames_geocode as gg
from imprints import marc_place_geonames as mpg


def _geonames_line(
    geonameid,
    name,
    asciiname=None,
    alternatenames="",
    lat=0.0,
    lon=0.0,
    feature_class="P",
    country_code="US",
    admin1_code="",
    population=0,
):
    """Build one tab-separated GeoNames main-dump line (19 columns)."""
    fields = [
        str(geonameid),
        name,
        asciiname if asciiname is not None else name,
        alternatenames,
        str(lat),
        str(lon),
        feature_class,
        "PPL",
        country_code,
        "",
        admin1_code,
        "",
        "",
        "",
        str(population),
        "",
        "",
        "America/New_York",
        "2020-01-01",
    ]
    return "\t".join(fields)


def _write_country_file(tmp_path, name, lines):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_load_geonames_index_keeps_only_populated_places(tmp_path):
    lines = [
        _geonames_line(
            1, "Athens", country_code="US", admin1_code="GA", population=100
        ),
        _geonames_line(
            2, "Georgia", country_code="US", admin1_code="GA", feature_class="A"
        ),
    ]
    path = _write_country_file(tmp_path, "US.txt", lines)
    index = gg.load_geonames_index([path])
    assert ("US", "athens") in index["primary"]
    assert ("US", "georgia") not in index["primary"]


def test_match_place_exact_within_scope(tmp_path):
    lines = [
        _geonames_line(
            1, "Athens", country_code="US", admin1_code="GA", population=100000
        )
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    result = gg.match_place("athens", ("US", "GA"), index)
    assert result["geonames_id"] == "1"
    assert result["geonames_admin1_code"] == "GA"
    assert result["geonames_ambiguous"] is False


def test_match_place_wrong_admin1_scope_does_not_match(tmp_path):
    lines = [
        _geonames_line(
            1, "Athens", country_code="US", admin1_code="GA", population=100000
        )
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    assert gg.match_place("athens", ("US", "OH"), index) is None


def test_match_place_prefers_primary_name_over_stale_alternate(tmp_path):
    """Regression: Lemont, IL carries "Athens" as a historical alternate
    name with a far higher population than the real Athens, IL. A merged
    name/alternate index would let Lemont win the population tie-break;
    the tiered index must prefer Athens' own primary-name match instead."""
    lines = [
        _geonames_line(
            1, "Athens", country_code="US", admin1_code="IL", population=1938
        ),
        _geonames_line(
            2,
            "Lemont",
            alternatenames="Athens,Palmyra",
            country_code="US",
            admin1_code="IL",
            population=16788,
        ),
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    result = gg.match_place("athens", ("US", "IL"), index)
    assert result["geonames_id"] == "1"
    assert result["geonames_name"] == "Athens"


def test_match_place_falls_back_to_alternate_when_no_primary_match(tmp_path):
    lines = [
        _geonames_line(
            1,
            "New York City",
            alternatenames="New York,NYC",
            country_code="US",
            admin1_code="NY",
            population=8000000,
        )
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    result = gg.match_place("new york", ("US", "NY"), index)
    assert result["geonames_id"] == "1"


def test_match_place_ambiguous_ties_broken_by_population(tmp_path):
    lines = [
        _geonames_line(
            1, "Franklin", country_code="US", admin1_code="OH", population=500
        ),
        _geonames_line(
            2, "Franklin", country_code="US", admin1_code="OH", population=12000
        ),
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    result = gg.match_place("franklin", ("US", "OH"), index)
    assert result["geonames_id"] == "2"
    assert result["geonames_ambiguous"] is True


def test_match_place_retries_with_canonicalized_city_after_dropping_state_token(
    tmp_path,
):
    """places_clean is frequently already in "<city> <postal>" form (e.g.
    "waterville me"), which GeoNames indexes as bare "Waterville" -- the
    retry must fire even though canonicalize_place("waterville me") returns
    the same string it was given."""
    lines = [
        _geonames_line(
            1, "Waterville", country_code="US", admin1_code="ME", population=5000
        )
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    result = gg.match_place("waterville me", ("US", "ME"), index)
    assert result["geonames_id"] == "1"


def test_match_place_no_candidate_returns_none(tmp_path):
    lines = [
        _geonames_line(1, "Athens", country_code="US", admin1_code="GA", population=100)
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    assert gg.match_place("nonexistent town", ("US", "GA"), index) is None
    assert gg.match_place("", ("US", "GA"), index) is None


def _admin1_fixture(tmp_path):
    path = tmp_path / "admin1CodesASCII.txt"
    path.write_text("US.GA\tGeorgia\tGeorgia\t1\nCA.10\tQuebec\tQuebec\t2\n")
    return str(path)


def test_resolve_llm_scope_us_city_state(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA"])
    scope, city = gg._resolve_llm_scope("athens, georgia", admin1_names, ["US", "CA"])
    assert scope == ("US", "GA")
    assert city == "athens"


def test_resolve_llm_scope_country_only(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA"])
    scope, city = gg._resolve_llm_scope("toronto, canada", admin1_names, ["US", "CA"])
    assert scope == ("CA", None)
    assert city == "toronto"


def test_resolve_llm_scope_prefers_admin1_over_country(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA"])
    scope, city = gg._resolve_llm_scope(
        "quebec city, quebec, canada", admin1_names, ["US", "CA"]
    )
    assert scope == ("CA", "10")
    assert city == "quebec city"


def test_resolve_llm_scope_unresolvable_returns_none_scope(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA"])
    scope, city = gg._resolve_llm_scope("paris, france", admin1_names, ["US", "CA"])
    assert scope is None
    assert city == "paris"
