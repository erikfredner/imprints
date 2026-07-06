"""
Test whether NYC-share findings are sensitive to how multi-place records are
weighted. Compares three treatments: dropping multi-place records, splitting
their weight evenly across places, and counting every place equally (the
existing repo default).

Usage:
    python -m imprints.multi_place_sensitivity --input-csv data/PS/data.csv
"""

import argparse

import pandas as pd

NO_PLACE = "No place of publication"
METHODS = ["drop", "fractional", "equal"]
METHOD_LABELS = {
    "drop": "1. Drop multi-place records",
    "fractional": "2. Split weight evenly (1/k)",
    "equal": "3. Count every place equally (default)",
}


def load_data(input_csv: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(input_csv)


def drop_missing_lccn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with a missing lccn. lccn is the only record identifier
    available, so without it a place row can't be reliably grouped back to
    its source record (pandas groupby silently drops NaN keys, which would
    otherwise make such records invisible to compute_place_counts and corrupt
    the "drop"/"fractional" weighting).
    """
    missing = df["lccn"].isna()
    if missing.any():
        print(
            f"Dropping {missing.sum():,} rows with missing lccn (no record id to group by).\n"
        )
    return df[~missing]


def compute_place_counts(df: pd.DataFrame) -> pd.Series:
    """Number of real (non-no-place) place rows per lccn, indexed by lccn."""
    is_real_place = df["city_group"] != NO_PLACE
    return is_real_place.groupby(df["lccn"]).sum()


def print_place_count_table(place_counts: pd.Series) -> None:
    """Print counts of PS records by number of places of publication."""
    total = len(place_counts)

    def bucket(n):
        if n == 0:
            return "No place of publication"
        if n == 1:
            return "Exactly one"
        return "More than one"

    counts = place_counts.map(bucket).value_counts()
    order = ["No place of publication", "Exactly one", "More than one"]

    print("PS records by number of places of publication")
    print("-" * 60)
    print(f"{'Category':<28}{'Records':>12}{'% of total':>14}")
    for label in order:
        n = int(counts.get(label, 0))
        pct = (n / total * 100) if total else 0.0
        print(f"{label:<28}{n:>12,}{pct:>13.2f}%")
    print(f"{'Total':<28}{total:>12,}")
    print()


def compute_city_share(
    df: pd.DataFrame,
    place_counts: pd.Series,
    method: str,
    city: str,
    start_year: int,
    end_year: int,
) -> pd.Series:
    """
    Compute annual percentage of weighted records published in `city`, under
    one of three treatments of multi-place records:
      - "drop": records with >1 place are excluded entirely (weight 1)
      - "fractional": each place on a k-place record gets weight 1/k
      - "equal": every place gets weight 1 (status quo, unweighted rows)
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method!r}")

    df = df.copy()
    df = df[df["year_min"].between(start_year, end_year)]

    if method == "drop":
        keep_lccn = place_counts[place_counts <= 1].index
        df = df[df["lccn"].isin(keep_lccn)]
        weight = pd.Series(1.0, index=df.index)
    elif method == "fractional":
        record_place_count = df["lccn"].map(place_counts)
        weight = 1.0 / record_place_count.where(record_place_count > 0, 1)
    else:  # "equal"
        weight = pd.Series(1.0, index=df.index)

    df = df[df["city_group"] != NO_PLACE]
    weight = weight.loc[df.index]

    in_city = (df["city_group"] == city).astype(int)
    weighted = pd.DataFrame(
        {"year_min": df["year_min"], "in_city": in_city, "weight": weight}
    )

    totals = (
        weighted.groupby(["year_min", "in_city"])["weight"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=[0, 1], fill_value=0.0)
        .sort_index()
    )
    pct = totals.div(totals.sum(axis=1), axis=0) * 100
    pct.index = pct.index.astype(int)
    return pct[1]


def compute_crossing_stats(pct_series: pd.Series) -> dict:
    """
    Replicates ps_nyc_share.py's crossing/peak logic: 50% thresholds use the rounded
    series; peak year/value use the raw series.
    """
    if pct_series.empty:
        return {
            "first_year_at_least_50": None,
            "last_year_at_least_50": None,
            "peak_year": None,
            "peak_pct": None,
        }

    pct_rounded = pct_series.round()
    at_least_50 = pct_rounded[pct_rounded >= 50]
    first_year_at_least_50 = (
        int(at_least_50.index.min()) if not at_least_50.empty else None
    )
    last_year_at_least_50 = (
        int(at_least_50.index.max()) if not at_least_50.empty else None
    )

    peak_year = int(pct_series.idxmax())
    peak_pct = float(pct_series.loc[peak_year])

    return {
        "first_year_at_least_50": first_year_at_least_50,
        "last_year_at_least_50": last_year_at_least_50,
        "peak_year": peak_year,
        "peak_pct": peak_pct,
    }


def _fmt(value, suffix=""):
    return "—" if value is None else f"{value}{suffix}"


def print_comparison_table(
    results: dict, city: str, start_year: int, end_year: int
) -> None:
    print(f"NYC share crossing/peak stats by method ({city}, {start_year}-{end_year})")
    print("-" * 100)
    header = (
        f"{'Method':<40}{'First yr >=50%':>16}{'Last yr >=50%':>16}"
        f"{'Peak yr':>10}{'Peak %':>10}"
    )
    print(header)
    for method in METHODS:
        stats = results[method]
        peak_pct = "—" if stats["peak_pct"] is None else f"{stats['peak_pct']:.2f}"
        print(
            f"{METHOD_LABELS[method]:<40}"
            f"{_fmt(stats['first_year_at_least_50']):>16}"
            f"{_fmt(stats['last_year_at_least_50']):>16}"
            f"{_fmt(stats['peak_year']):>10}"
            f"{peak_pct:>10}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Test sensitivity of NYC-share findings to how multi-place "
            "records are weighted (drop / fractional / equal-count)."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/PS/data.csv",
        help="Path to the cleaned PS data CSV (default: data/PS/data.csv).",
    )
    parser.add_argument(
        "--city",
        default="New York City",
        help="City to test; defaults to the 'New York City' label used in the data.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1900,
        help="Start year (inclusive, default: 1900).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2010,
        help="End year (inclusive, default: 2010).",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    df = drop_missing_lccn(df)
    place_counts = compute_place_counts(df)

    print_place_count_table(place_counts)

    results = {}
    for method in METHODS:
        pct_series = compute_city_share(
            df,
            place_counts,
            method=method,
            city=args.city,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        results[method] = compute_crossing_stats(pct_series)

    print_comparison_table(results, args.city, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
