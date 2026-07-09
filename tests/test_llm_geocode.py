"""Offline tests for the LLM residual-selection helpers."""

import pandas as pd

from imprints import llm_geocode as lg


def test_load_geo_keys_skips_only_successful_direct_matches(tmp_path):
    path = tmp_path / "direct.csv"
    pd.DataFrame(
        {
            "geo_key": ["matched", "unmatched"],
            "geonames_matched": [True, False],
        }
    ).to_csv(path, index=False)

    assert lg._load_geo_keys(str(path)) == {"matched"}


def test_load_geo_keys_accepts_pre_filtered_legacy_list(tmp_path):
    path = tmp_path / "keys.csv"
    pd.DataFrame({"geo_key": ["one", "two"]}).to_csv(path, index=False)

    assert lg._load_geo_keys(str(path)) == {"one", "two"}
