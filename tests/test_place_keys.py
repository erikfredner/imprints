"""Tests for imprints.place_keys."""

import pandas as pd

from imprints import place_keys as pk


def test_build_geo_key_combines_places_clean_and_place_name_008():
    assert pk.build_geo_key("athens", "Georgia") == "athens||Georgia"
    assert pk.build_geo_key("athens", "Ohio") == "athens||Ohio"
    assert pk.build_geo_key("athens", "Georgia") != pk.build_geo_key("athens", "Ohio")


def test_build_geo_key_missing_008_matches_places_clean_alone():
    assert pk.build_geo_key("boston", None) == pk.build_geo_key("boston", "")
    assert pk.build_geo_key("boston", None) == pk.build_geo_key("boston", float("nan"))
    assert pk.build_geo_key("boston", pd.NA) == "boston||"


def test_build_geo_key_normalizes_none_places_clean():
    assert pk.build_geo_key(None, None) == "||"


def test_build_geo_key_separates_nyc_multiplace_candidate():
    normal = pk.build_geo_key("new york", "England")
    candidate = pk.build_geo_key(
        "new york",
        "England",
        pk.GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE,
    )
    assert candidate == "new york||England||nyc_multiplace_candidate"
    assert candidate != normal


def test_add_geocode_key_columns_marks_only_multiplace_nyc_component():
    df = pd.DataFrame(
        [
            {
                "lccn": "multi",
                "places_clean": "new york",
                "place_name_008": "England",
                "city_group": "New York City",
            },
            {
                "lccn": "multi",
                "places_clean": "london",
                "place_name_008": "England",
                "city_group": "Other",
            },
            {
                "lccn": "single",
                "places_clean": "new york",
                "place_name_008": "England",
                "city_group": "New York City",
            },
        ]
    )

    out = pk.add_geocode_key_columns(df)

    assert out.loc[0, "geo_key_policy"] == pk.GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE
    assert out.loc[0, "geo_key"] == "new york||England||nyc_multiplace_candidate"
    assert out.loc[1, "geo_key_policy"] == pk.GEO_KEY_POLICY_008
    assert out.loc[1, "geo_key"] == "london||England"
    assert out.loc[2, "geo_key_policy"] == pk.GEO_KEY_POLICY_008
    assert out.loc[2, "geo_key"] == "new york||England"


def test_build_place_hint_with_008_only():
    assert pk.build_place_hint("Georgia") == "marc_008_place_hint: Georgia"


def test_build_place_hint_with_008_and_752():
    hint = pk.build_place_hint("Ohio", "United States, Ohio, Athens")
    assert hint == (
        "marc_008_place_hint: Ohio\nmarc_752_place_hint: United States, Ohio, Athens"
    )


def test_build_place_hint_none_when_no_signal():
    assert pk.build_place_hint(None, None) is None
    assert pk.build_place_hint("", "") is None
    assert pk.build_place_hint(float("nan"), None) is None
