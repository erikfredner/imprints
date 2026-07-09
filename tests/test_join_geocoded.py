"""Tests for joining policy-aware GeoNames results."""

import pandas as pd

from imprints import join_geocoded as jg
from imprints import place_keys as pk


def _result(
    geo_key,
    *,
    matched,
    lat=None,
    lon=None,
    country=None,
    policy=None,
    reason=None,
):
    return {
        "geo_key": geo_key,
        "geonames_matched": matched,
        "geonames_lat": lat,
        "geonames_lon": lon,
        "geonames_country_code": country,
        "geocode_policy": policy,
        "geocode_reason": reason,
    }


def test_join_preserves_nyc_multiplace_override_policy():
    data = pk.add_geocode_key_columns(
        pd.DataFrame(
            [
                {
                    "lccn": "multi",
                    "year_min": 2000,
                    "places_clean": "new york",
                    "place_name_008": "England",
                    "city_group": "New York City",
                },
                {
                    "lccn": "multi",
                    "year_min": 2000,
                    "places_clean": "london",
                    "place_name_008": "England",
                    "city_group": "Other",
                },
                {
                    "lccn": "single",
                    "year_min": 2000,
                    "places_clean": "new york",
                    "place_name_008": "England",
                    "city_group": "New York City",
                },
            ]
        )
    )
    candidate_key = pk.build_geo_key(
        "new york",
        "England",
        pk.GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE,
    )
    direct = pd.DataFrame(
        [
            _result(
                candidate_key,
                matched=True,
                lat=40.7128,
                lon=-74.006,
                country="US",
                policy="nyc_multiplace_override",
                reason="multi-place NYC label overrides 'England' 008 scope",
            ),
            _result(
                "london||England",
                matched=True,
                lat=51.5072,
                lon=-0.1276,
                country="GB",
                policy="direct_008",
            ),
            _result(
                "new york||England",
                matched=True,
                lat=53.07897,
                lon=-0.14008,
                country="GB",
                policy="direct_008",
            ),
        ]
    )
    llm = pd.DataFrame(columns=jg.GEONAMES_COLUMNS)

    out = jg.join(data, llm, direct)

    assert list(out["geocode_source"]) == [
        "geonames_direct",
        "geonames_direct",
        "geonames_direct",
    ]
    assert list(out["geocode_policy"]) == [
        "nyc_multiplace_override",
        "direct_008",
        "direct_008",
    ]
    assert out.loc[0, "geocode_reason"] == (
        "multi-place NYC label overrides 'England' 008 scope"
    )
    assert list(out["geocoded_country_code"]) == ["us", "gb", "gb"]


def test_load_geonames_accepts_legacy_results_without_policy(tmp_path):
    path = tmp_path / "legacy.csv"
    pd.DataFrame(
        [
            _result(
                "athens||Georgia",
                matched=True,
                lat=33.96,
                lon=-83.37,
                country="US",
            )
        ]
    ).drop(columns="geocode_policy").to_csv(path, index=False)

    loaded = jg.load_geonames(str(path))

    assert list(loaded.columns) == jg.GEONAMES_COLUMNS
    assert pd.isna(loaded.loc[0, "geocode_policy"])
