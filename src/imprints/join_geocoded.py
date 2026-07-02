"""
Join cleaned PS records with their geocoded places of publication.

Left-merges `data/PS/data.csv` (record-level, one row per place of
publication) onto `data/PS/llm_geocode_nominatim.csv` (one row per unique
`places_clean` value, produced by `imprints.geocode_sample llm` from
`imprints.llm_geocode`'s output) on the `places_clean` join key. The two are
keyed identically -- `geocode_sample` reads `places_clean` straight out of
`data.csv` without further normalization -- so this is a plain exact-match
merge, many-to-one from data.csv's side.

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

OUTPUT_COLUMNS = [
    "lccn",
    "year_min",
    "places_clean",
    "city_group",
    "llm_nominatim_lat",
    "llm_nominatim_lon",
    "llm_nominatim_country_code",
]


def load_data(input_csv: str) -> pd.DataFrame:
    """Load the record-level columns needed for the join and downstream use."""
    return pd.read_csv(
        input_csv, usecols=["lccn", "year_min", "places_clean", "city_group"]
    )


def load_nominatim(nominatim_csv: str) -> pd.DataFrame:
    """Load the per-unique-place LLM-normalized geocoding results."""
    df = pd.read_csv(
        nominatim_csv,
        usecols=[
            "places_clean",
            "llm_nominatim_found",
            "llm_nominatim_country_code",
            "llm_nominatim_lat",
            "llm_nominatim_lon",
        ],
    )
    if df["places_clean"].duplicated().any():
        raise ValueError(
            f"{nominatim_csv} has duplicate places_clean values; expected "
            "one row per unique place."
        )
    return df


def join(data_df: pd.DataFrame, nominatim_df: pd.DataFrame) -> pd.DataFrame:
    """Left-merge data_df onto nominatim_df on places_clean, keeping every
    data_df row (including unmatched/non-US ones)."""
    merged = data_df.merge(nominatim_df, on="places_clean", how="left")
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
