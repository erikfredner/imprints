"""
Join cleaned PS records with their geocoded places of publication.

Left-merges `data/PS/data.csv` (record-level, one row per place of
publication) onto `data/PS/llm_geocode_nominatim.csv` (one row per
`geo_key` group, produced by `imprints.geocode_sample llm` from
`imprints.llm_geocode`'s output) on the `geo_key` join key --
`places_clean` plus the decoded MARC 008 place-of-publication code
(`place_name_008`), via `imprints.place_keys.build_geo_key`. Both sides
build the key identically from the same two source columns, so this is a
plain exact-match merge, many-to-one from data.csv's side. Grouping on
`places_clean` alone would conflate records that share an ambiguous bare
city name (e.g. "Athens") but differ in 008 place code -- and therefore in
true location.

Coordinates come from the `llm_nominatim_*` columns: Nominatim results for
the LLM-normalized place name, not the raw `places_clean` string.

All rows from data.csv are kept, including those with no Nominatim match or
a non-US result; scope-specific filtering (e.g. to US-only records) is left
to whichever figure or analysis consumes this output.

Usage:
    python -m imprints.join_geocoded \
        --input-csv data/PS/data.csv \
        --nominatim-csv data/PS/llm_geocode_nominatim.csv \
        --output-csv data/PS/geocoded.csv
"""

import argparse

import pandas as pd

from imprints import place_keys

OUTPUT_COLUMNS = [
    "lccn",
    "year_min",
    "places_clean",
    "place_name_008",
    "city_group",
    "llm_nominatim_lat",
    "llm_nominatim_lon",
    "llm_nominatim_country_code",
]


def load_data(input_csv: str) -> pd.DataFrame:
    """Load the record-level columns needed for the join and downstream use,
    and compute geo_key. Missing place_name_008 (older data.csv, pre-008
    capture) is treated as all-missing, so the key falls back to
    places_clean alone -- consistent with imprints.geocode_sample."""
    header = pd.read_csv(input_csv, nrows=0)
    wanted = ["lccn", "year_min", "places_clean", "city_group", "place_name_008"]
    usecols = [c for c in wanted if c in header.columns]
    df = pd.read_csv(input_csv, usecols=usecols)
    if "place_name_008" not in df.columns:
        df["place_name_008"] = None
    df["geo_key"] = [
        place_keys.build_geo_key(pc, p8)
        for pc, p8 in zip(df["places_clean"], df["place_name_008"])
    ]
    return df


def load_nominatim(nominatim_csv: str) -> pd.DataFrame:
    """Load the per-geo_key-group LLM-normalized geocoding results."""
    df = pd.read_csv(
        nominatim_csv,
        usecols=[
            "geo_key",
            "llm_nominatim_found",
            "llm_nominatim_country_code",
            "llm_nominatim_lat",
            "llm_nominatim_lon",
        ],
    )
    if df["geo_key"].duplicated().any():
        raise ValueError(
            f"{nominatim_csv} has duplicate geo_key values; expected one "
            "row per unique (places_clean, place_name_008) group."
        )
    return df


def join(data_df: pd.DataFrame, nominatim_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge data_df onto nominatim_df on geo_key, keeping every
    data_df row (including unmatched/non-US ones)."""
    merged = data_df.merge(nominatim_df, on="geo_key", how="left")
    return merged[OUTPUT_COLUMNS]


def print_summary(merged: pd.DataFrame) -> None:
    """Print match-rate diagnostics for the joined table."""
    total = len(merged)
    matched = merged["llm_nominatim_lat"].notna().sum()
    us = (merged["llm_nominatim_country_code"] == "us").sum()
    print(f"Joined rows: {total:,}")
    print(f"Rows with a Nominatim match: {matched:,} ({matched / total:.1%})")
    print(
        f"Rows resolving to US (llm_nominatim_country_code == 'us'): {us:,} ({us / total:.1%})"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Join cleaned PS records with Nominatim geocoding results "
        "on places_clean."
    )
    parser.add_argument(
        "--input-csv",
        default="data/PS/data.csv",
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--nominatim-csv",
        default="data/PS/llm_geocode_nominatim.csv",
        help="Path to LLM-normalized Nominatim geocoding results CSV "
        "(default: data/PS/llm_geocode_nominatim.csv)",
    )
    parser.add_argument(
        "--output-csv",
        default="data/PS/geocoded.csv",
        help="Output path for the joined CSV (default: data/PS/geocoded.csv)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input_csv}")
    data_df = load_data(args.input_csv)
    print(f"Loading {args.nominatim_csv}")
    nominatim_df = load_nominatim(args.nominatim_csv)

    merged = join(data_df, nominatim_df)
    print_summary(merged)

    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
