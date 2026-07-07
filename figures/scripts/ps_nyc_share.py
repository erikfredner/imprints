#!/usr/bin/env python3
"""
Generate a line plot showing the percentage of PS-class imprints published in
New York City vs. other locations over time.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/ps_nyc_share.png"

#: ``city_group`` value for records lacking a place of publication. Dropped so the
#: NYC share is NYC / (NYC + Other), matching the other figures.
NO_PLACE = "No place of publication"


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def compute_city_share(
    df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
    window: int = 5,
    smooth: bool = False,
    include_ci: bool = False,
) -> pd.DataFrame:
    """
    Compute annual percentage of records published in `city`.
    Optionally smooths counts with a rolling window before converting to percentages.
    """
    df = df.copy()
    df = df[df["year_min"].between(start_year, end_year)]
    df = df[df["city_group"] != NO_PLACE]
    df["in_city"] = (df["city_group"] == city).astype(int)
    window = max(1, window)

    counts = (
        df.groupby(["year_min", "in_city"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .sort_index()
    )
    counts_raw = counts.copy()

    if smooth:
        counts = counts.rolling(window=window, min_periods=1).mean()

    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    pct.index = pct.index.astype(int)

    if include_ci:
        totals = counts_raw.sum(axis=1).replace(0, pd.NA)
        p = counts_raw[1] / totals
        se = (p * (1 - p) / totals) ** 0.5
        z = 1.96
        pct["ci_lower"] = (p - z * se).mul(100).clip(lower=0)
        pct["ci_upper"] = (p + z * se).mul(100).clip(upper=100)
    return pct


def plot_share(
    pct_df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
    output_path: str = None,
    include_ci: bool = False,
) -> None:
    """Plot a line chart of city's share of records over time and save or show."""
    style.apply_style()
    fig, ax = plt.subplots()
    pct_city = pct_df[1]
    years = pct_city.index.values

    if include_ci and "ci_lower" in pct_df.columns and "ci_upper" in pct_df.columns:
        lower = pct_df["ci_lower"]
        upper = pct_df["ci_upper"]
        ax.fill_between(years, lower, upper, color=style.COLOR_NYC, alpha=0.2)

    ax.plot(years, pct_city, color=style.COLOR_NYC)
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of PS works published in " + city)
    style.percent_yaxis(ax)
    plt.tight_layout()
    if output_path:
        style.save_figure(output_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot PS imprints published in New York vs. other locations"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--city",
        default="New York City",
        help="City to highlight; defaults to the 'New York City' label used in the data",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1900,
        help="Start year (inclusive, default: 1900)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2010,
        help="End year (inclusive, default: 2010)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window size in years for smoothing (default: 5)",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply rolling smoothing to annual percentages (default: false)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: figures/outputs/ps_nyc_share.png)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        default=False,
        help="Include 95 pct confidence intervals for city share",
    )
    args = parser.parse_args()

    city = "New York City" if args.city == "New York" else args.city

    df = load_data(args.input_csv)
    pct = compute_city_share(
        df,
        city=city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        smooth=args.smooth,
        include_ci=args.ci,
    )

    pct_city = pct[1]
    pct_rounded = pct_city.round()

    first_at_least_50 = pct_rounded[pct_rounded >= 50]
    year_first_at_least_50 = (
        int(first_at_least_50.index.min()) if not first_at_least_50.empty else None
    )

    year_first_below_after_50 = None
    if year_first_at_least_50 is not None:
        after_first = pct_rounded.loc[pct_rounded.index > year_first_at_least_50]
        below_after = after_first[after_first < 50]
        if not below_after.empty:
            year_first_below_after_50 = int(below_after.index.min())

    year_lowest = int(pct_city.idxmin()) if not pct_city.empty else None
    pct_lowest = float(pct_city.loc[year_lowest]) if year_lowest is not None else None
    year_highest = int(pct_city.idxmax()) if not pct_city.empty else None
    pct_highest = (
        float(pct_city.loc[year_highest]) if year_highest is not None else None
    )

    print("First year NYC share >= 50% (rounded):", year_first_at_least_50)
    print(
        "First year NYC share falls below 50% after crossing:",
        year_first_below_after_50,
    )
    print("Year with lowest NYC share:", year_lowest)
    print(
        "NYC share in that year (%):",
        None if pct_lowest is None else round(pct_lowest, 2),
    )
    print("Year with highest NYC share:", year_highest)
    print(
        "NYC share in that year (%):",
        None if pct_highest is None else round(pct_highest, 2),
    )

    plot_share(
        pct,
        city=city,
        start_year=args.start_year,
        end_year=args.end_year,
        output_path=args.output,
        include_ci=args.ci,
    )


if __name__ == "__main__":
    main()
