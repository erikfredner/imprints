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
the LLM-normalized place name, not the raw `places_clean` string -- unless
`--geonames-direct-csv` is given (default: `data/PS/geonames_direct.csv`, if
present), in which case a geo_key resolved by `imprints.geonames_geocode`
direct mode (`geonames_matched=True`) uses *those* coordinates/country code
instead, since that pathway matches the record's own place_name_008-scoped
gazetteer entry rather than an LLM guess. The column names stay
`llm_nominatim_*` either way (so this stays a drop-in replacement for
existing consumers, e.g. `figures/scripts/nyc_peak_map*.py`, with no
changes needed there); a new `geocode_source` column
(`geonames_direct`/`llm_nominatim`/`unmatched`) records which pathway
actually produced each row's coordinates. Pass `--geonames-direct-csv ""` to
disable this and reproduce the old LLM-only behavior exactly.

All rows from data.csv are kept, including those with no match or a non-US
result; scope-specific filtering (e.g. to US-only records) is left to
whichever figure or analysis consumes this output.

Usage:
    python -m imprints.join_geocoded \
        --input-csv data/PS/data.csv \
        --nominatim-csv data/PS/llm_geocode_nominatim.csv \
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


NOMINATIM_COLUMNS = [
    "geo_key",
    "llm_nominatim_found",
    "llm_nominatim_country_code",
    "llm_nominatim_lat",
    "llm_nominatim_lon",
]


def load_nominatim(nominatim_csv: str) -> pd.DataFrame:
    """Load the per-geo_key-group LLM-normalized geocoding results.

    Returns an empty (but correctly shaped) DataFrame, with a printed
    warning, if nominatim_csv predates the geo_key key (see
    imprints.place_keys) -- e.g. an old data/PS/llm_geocode_nominatim.csv
    generated before the 008-disambiguation migration -- rather than
    crashing. Records that would have relied on it simply have no
    LLM+Nominatim fallback available and are left unmatched by that source;
    imprints.geonames_geocode's direct pathway does not depend on this
    file at all."""
    header = pd.read_csv(nominatim_csv, nrows=0)
    if "geo_key" not in header.columns:
        print(
            f"WARNING: {nominatim_csv} has no geo_key column (predates the "
            "geo_key migration) -- proceeding with no LLM+Nominatim "
            "fallback data. See imprints.geocode_sample's module docstring "
            "for the migration recipe."
        )
        return pd.DataFrame(columns=NOMINATIM_COLUMNS)

    df = pd.read_csv(nominatim_csv, usecols=NOMINATIM_COLUMNS)
    if df["geo_key"].duplicated().any():
        raise ValueError(
            f"{nominatim_csv} has duplicate geo_key values; expected one "
            "row per unique (places_clean, place_name_008) group."
        )
    return df


def load_geonames_direct(geonames_direct_csv: str) -> pd.DataFrame:
    """Load the per-geo_key-group direct GeoNames matching results
    (imprints.geonames_geocode direct mode)."""
    df = pd.read_csv(
        geonames_direct_csv,
        usecols=[
            "geo_key",
            "geonames_matched",
            "geonames_lat",
            "geonames_lon",
            "geonames_country_code",
        ],
    )
    if df["geo_key"].duplicated().any():
        raise ValueError(
            f"{geonames_direct_csv} has duplicate geo_key values; expected "
            "one row per unique (places_clean, place_name_008) group."
        )
    return df


def join(
    data_df: pd.DataFrame,
    nominatim_df: pd.DataFrame,
    geonames_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Left-merge data_df onto nominatim_df (and, if given, geonames_df) on
    geo_key, keeping every data_df row (including unmatched/non-US ones).

    When geonames_df is given, a geo_key it resolved (geonames_matched=True)
    has its llm_nominatim_lat/lon/country_code columns overwritten with the
    GeoNames direct match instead -- see module docstring. geocode_source
    records which pathway won for each row."""
    merged = data_df.merge(nominatim_df, on="geo_key", how="left")

    if geonames_df is None:
        merged["geocode_source"] = (
            merged["llm_nominatim_lat"]
            .notna()
            .map({True: "llm_nominatim", False: "unmatched"})
        )
        return merged[OUTPUT_COLUMNS + ["geocode_source"]]

    merged = merged.merge(geonames_df, on="geo_key", how="left")
    use_geonames = merged["geonames_matched"].fillna(False)

    merged["geocode_source"] = "unmatched"
    merged.loc[merged["llm_nominatim_lat"].notna(), "geocode_source"] = "llm_nominatim"
    merged.loc[use_geonames, "geocode_source"] = "geonames_direct"

    merged["llm_nominatim_lat"] = merged["geonames_lat"].where(
        use_geonames, merged["llm_nominatim_lat"]
    )
    merged["llm_nominatim_lon"] = merged["geonames_lon"].where(
        use_geonames, merged["llm_nominatim_lon"]
    )
    merged["llm_nominatim_country_code"] = (
        merged["geonames_country_code"]
        .str.lower()
        .where(use_geonames, merged["llm_nominatim_country_code"])
    )
    return merged[OUTPUT_COLUMNS + ["geocode_source"]]


def print_summary(merged: pd.DataFrame) -> None:
    """Print match-rate diagnostics for the joined table."""
    total = len(merged)
    matched = merged["llm_nominatim_lat"].notna().sum()
    us = (merged["llm_nominatim_country_code"] == "us").sum()
    print(f"Joined rows: {total:,}")
    print(f"Rows with a match: {matched:,} ({matched / total:.1%})")
    print(
        f"Rows resolving to US (llm_nominatim_country_code == 'us'): {us:,} ({us / total:.1%})"
    )
    print("By source:")
    for source, count in merged["geocode_source"].value_counts().items():
        print(f"  {source}: {count:,} ({count / total:.1%})")


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
        "--geonames-direct-csv",
        default="data/PS/geonames_direct.csv",
        help="Path to imprints.geonames_geocode direct mode's output. When "
        "present, a geo_key it resolved wins over the LLM+Nominatim result "
        "for that geo_key (see module docstring). Pass an empty string to "
        "disable and use LLM+Nominatim results only.",
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

    geonames_df = None
    if args.geonames_direct_csv and os.path.exists(args.geonames_direct_csv):
        print(f"Loading {args.geonames_direct_csv}")
        geonames_df = load_geonames_direct(args.geonames_direct_csv)
    elif args.geonames_direct_csv:
        print(
            f"{args.geonames_direct_csv} not found, skipping GeoNames direct results."
        )

    merged = join(data_df, nominatim_df, geonames_df)
    print_summary(merged)

    merged.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(merged):,} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
