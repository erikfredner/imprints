"""Tests for imprints.geonames_geocode."""

import pandas as pd

from imprints import geonames_geocode as gg
from imprints import marc_place_geonames as mpg
from imprints import place_keys as pk


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


def test_direct_multiplace_nyc_component_overrides_conflicting_008_scope(tmp_path):
    """A record-level England 008 applies to London, not its NYC co-place.

    The NYC component gets a distinct candidate key and is resolved to the
    canonical NYC coordinate. A single-place NYC-labelled row sharing the old
    ``new york||England`` key remains an ordinary England-scoped match.
    """
    us = _write_country_file(
        tmp_path,
        "US.txt",
        [
            _geonames_line(
                1,
                "New York City",
                alternatenames="New York,NYC",
                country_code="US",
                admin1_code="NY",
                population=8000000,
            )
        ],
    )
    gb = _write_country_file(
        tmp_path,
        "GB.txt",
        [
            _geonames_line(
                2,
                "New York",
                country_code="GB",
                admin1_code="ENG",
                population=100,
            ),
            _geonames_line(
                3,
                "London",
                country_code="GB",
                admin1_code="ENG",
                population=9000000,
            ),
        ],
    )
    index = gg.load_geonames_index([us, gb])

    input_csv = tmp_path / "input.csv"
    pd.DataFrame(
        [
            {
                "lccn": "multi",
                "places": "New York",
                "places_clean": "new york",
                "place_name_008": "England",
                "city_group": "New York City",
            },
            {
                "lccn": "multi",
                "places": "London",
                "places_clean": "london",
                "place_name_008": "England",
                "city_group": "Other",
            },
            {
                "lccn": "single",
                "places": "New York",
                "places_clean": "new york",
                "place_name_008": "England",
                "city_group": "New York City",
            },
        ]
    ).to_csv(input_csv, index=False)
    crosswalk = tmp_path / "crosswalk.csv"
    crosswalk.write_text(
        "place_name_008,geonames_country_code,geonames_admin1_code\n"
        "England,GB,ENG\n"
        "New York (State),US,NY\n"
    )
    output_csv = tmp_path / "direct.csv"

    gg.run_direct(str(input_csv), str(crosswalk), index, str(output_csv))
    out = pd.read_csv(output_csv).set_index("geo_key")

    candidate_key = pk.build_geo_key(
        "new york",
        "England",
        pk.GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE,
    )
    assert str(out.loc[candidate_key, "geonames_id"]) == "1"
    assert out.loc[candidate_key, "geocode_policy"] == "nyc_multiplace_override"
    assert "overrides 'England'" in out.loc[candidate_key, "geocode_reason"]

    # The other co-publication component still follows the 008 scope.
    assert str(out.loc["london||England", "geonames_id"]) == "3"
    assert out.loc["london||England", "geocode_policy"] == "direct_008"

    # A single-place row is not changed merely because its city label is NYC.
    assert str(out.loc["new york||England", "geonames_id"]) == "2"
    assert out.loc["new york||England", "geocode_policy"] == "direct_008"


def test_direct_multiplace_nyc_candidate_keeps_correct_new_york_008_scope(tmp_path):
    lines = [
        _geonames_line(
            1,
            "New York City",
            alternatenames="New York",
            country_code="US",
            admin1_code="NY",
            population=8000000,
        )
    ]
    index = gg.load_geonames_index([_write_country_file(tmp_path, "US.txt", lines)])
    crosswalk = tmp_path / "crosswalk.csv"
    crosswalk.write_text(
        "place_name_008,geonames_country_code,geonames_admin1_code\n"
        "New York (State),US,NY\n"
    )
    row = pd.Series(
        {
            "places_clean": "new york",
            "place_name_008": "New York (State)",
            "geo_key_policy": pk.GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE,
        }
    )

    result, policy, reason = gg.resolve_direct_place(row, str(crosswalk), index)

    assert result["geonames_id"] == "1"
    assert policy == "direct_008"
    assert reason is None


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


def test_resolve_llm_scope_country_name_outside_downloaded_set(tmp_path):
    # "france" resolves through the country-name crosswalk even though FR is
    # not among the downloaded admin1 countries: the llm pass covers country
    # names place_name_008 never uses (see README, "GeoNames reference data").
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA"])
    scope, city = gg._resolve_llm_scope("paris, france", admin1_names, ["US", "CA"])
    assert scope == ("FR", None)
    assert city == "paris"


def test_resolve_llm_scope_unresolvable_returns_none_scope(tmp_path):
    admin1_names = mpg.load_admin1_names(_admin1_fixture(tmp_path), ["US", "CA"])
    scope, city = gg._resolve_llm_scope("x, atlantis", admin1_names, ["US", "CA"])
    assert scope is None
    assert city == "x"
