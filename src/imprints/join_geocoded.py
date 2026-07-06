"""
Join cleaned PS records with their geocoded places of publication.

Left-merges `data/PS/data.csv` (record-level, one row per place of
publication) onto the two GeoNames geocoding passes (`imprints.geonames_geocode`)
on the `geo_key` join key -- `places_clean` plus the decoded MARC 008
place-of-publication code (`place_name_008`), via
`imprints.place_keys.build_geo_key`. Both sides build the key identically
from the same two source columns, so this is a plain exact-match merge,
many-to-one from data.csv's side. Grouping on `places_clean` alone would
conflate records that share an ambiguous bare city name (e.g. "Athens") but
differ in 008 place code -- and therefore in true location.

Coordinates come from `--geonames-llm-csv` (default:
`data/PS/geonames_llm.csv`) -- the residual pathway that matches
`imprints.llm_geocode`'s LLM-normalized place strings against the GeoNames
gazetteer -- unless `--geonames-direct-csv` is given (default:
`data/PS/geonames_direct.csv`, if present), in which case a geo_key resolved
by `imprints.geonames_geocode` direct mode (`geonames_matched=True`) uses
*those* coordinates/country code instead, since that pathway matches the
record's own place_name_008-scoped gazetteer entry directly rather than via
an LLM guess. A `geocode_source` column (`geonames_direct`/`geonames_llm`/
`unmatched`) records which pathway actually produced each row's coordinates.
Pass `--geonames-direct-csv ""` to disable the direct-pass overlay and use
the residual pathway's results only.

All rows from data.csv are kept, including those with no match or a non-US
result; scope-specific filtering (e.g. to US-only records) is left to
whichever figure or analysis consumes this output.

Usage:
    python -m imprints.join_geocoded \
        --input-csv data/PS/data.csv \
        --geonames-llm-csv data/PS/geonames_llm.csv \
        --geonames-direct-csv data/PS/geonames_direct.csv \
        --output-csv data/PS/geocoded.csv
"""

import argparse
import os

import pandas as pd

from imprints import place_keys

OUTPUT_COLUMNS = [
    "lccn",
    "year_min",
    "places_clean",
    "place_name_008",
    "city_group",
    "geocoded_lat",
    "geocoded_lon",
    "geocoded_country_code",
]

GEONAMES_COLUMNS = [
    "geo_key",
    "geonames_matched",
    "geonames_lat",
    "geonames_lon",
    "geonames_country_code",
]


def load_data(input_csv: str) -> pd.DataFrame:
    """Load the record-level columns needed for the join and downstream use,
    and compute geo_key. Missing place_name_008 (older data.csv, pre-008
    capture) is treated as all-missing, so the key falls back to
    places_clean alone -- consistent with imprints.geonames_geocode."""
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


def load_geonames(geonames_csv: str) -> pd.DataFrame:
    """Load a per-geo_key-group GeoNames matching result (either
    imprints.geonames_geocode direct or llm mode output -- both share the
    same relevant columns)."""
    df = pd.read_csv(geonames_csv, usecols=GEONAMES_COLUMNS)
    if df["geo_key"].duplicated().any():
        raise ValueError(
            f"{geonames_csv} has duplicate geo_key values; expected one "
            "row per unique (places_clean, place_name_008) group."
        )
    return df


def join(
    data_df: pd.DataFrame,
    geonames_llm_df: pd.DataFrame,
    geonames_direct_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-merge data_df onto geonames_llm_df (and, if given,
    geonames_direct_df) on geo_key, keeping every data_df row (including
    unmatched/non-US ones).

    geonames_llm_df supplies the base coordinates (the residual pathway).
    When geonames_direct_df is given, a geo_key it resolved
    (geonames_matched=True) has its geocoded_lat/lon/country_code columns
    overwritten with the direct match instead -- see module docstring.
    geocode_source records which pathway won for each row."""
    merged = data_df.merge(geonames_llm_df, on="geo_key", how="left")

    llm_matched = merged["geonames_matched"].fillna(False)
    merged["geocode_source"] = llm_matched.map(
        {True: "geonames_llm", False: "unmatched"}
    )
    merged["geocoded_lat"] = merged["geonames_lat"]
    merged["geocoded_lon"] = merged["geonames_lon"]
    merged["geocoded_country_code"] = merged["geonames_country_code"].str.lower()
    merged = merged.drop(columns=GEONAMES_COLUMNS[1:])

    if geonames_direct_df is not None:
        merged = merged.merge(geonames_direct_df, on="geo_key", how="left")
        use_direct = merged["geonames_matched"].fillna(False)

        merged.loc[use_direct, "geocode_source"] = "geonames_direct"
        merged["geocoded_lat"] = merged["geonames_lat"].where(
            use_direct, merged["geocoded_lat"]
        )
        merged["geocoded_lon"] = merged["geonames_lon"].where(
            use_direct, merged["geocoded_lon"]
        )
        merged["geocoded_country_code"] = (
            merged["geonames_country_code"]
            .str.lower()
            .where(use_direct, merged["geocoded_country_code"])
        )

    return merged[OUTPUT_COLUMNS + ["geocode_source"]]


def print_summary(merged: pd.DataFrame) -> None:
    """Print match-rate diagnostics for the joined table."""
    total = len(merged)
    matched = merged["geocoded_lat"].notna().sum()
    us = (merged["geocoded_country_code"] == "us").sum()
    print(f"Joined rows: {total:,}")
    print(f"Rows with a match: {matched:,} ({matched / total:.1%})")
    print(
        f"Rows resolving to US (geocoded_country_code == 'us'): {us:,} ({us / total:.1%})"
    )
    print("By source:")
    for source, count in merged["geocode_source"].value_counts().items():
        print(f"  {source}: {count:,} ({count / total:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Join cleaned PS records with GeoNames geocoding results "
        "on places_clean."
    )
    parser.add_argument(
        "--input-csv",
        default="data/PS/data.csv",
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--geonames-llm-csv",
        default="data/PS/geonames_llm.csv",
        help="Path to imprints.geonames_geocode llm mode's output -- the "
        "residual pathway's geocoding results (default: "
        "data/PS/geonames_llm.csv)",
    )
    parser.add_argument(
        "--geonames-direct-csv",
        default="data/PS/geonames_direct.csv",
        help="Path to imprints.geonames_geocode direct mode's output. When "
        "present, a geo_key it resolved wins over the residual pathway's "
        "result for that geo_key (see module docstring). Pass an empty "
        "string to disable and use the residual pathway's results only.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/PS/geocoded.csv",
        help="Output path for the joined CSV (default: data/PS/geocoded.csv)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input_csv}")
    data_df = load_data(args.input_csv)
    print(f"Loading {args.geonames_llm_csv}")
    geonames_llm_df = load_geonames(args.geonames_llm_csv)

    geonames_direct_df = None
    if args.geonames_direct_csv and os.path.exists(args.geonames_direct_csv):
        print(f"Loading {args.geonames_direct_csv}")
        geonames_direct_df = load_geonames(args.geonames_direct_csv)
    elif args.geonames_direct_csv:
        print(
            f"{args.geonames_direct_csv} not found, skipping GeoNames direct results."
        )

    merged = join(data_df, geonames_llm_df, geonames_direct_df)
    print_summary(merged)

    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
