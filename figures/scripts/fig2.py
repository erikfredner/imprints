#!/usr/bin/env python3
"""
Generate an annual line chart of absolute counts of PS-class imprints
published in New York vs other locations (1900-2010).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig2.png"
YEAR_START = 1900
YEAR_END = 2010


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def compute_counts(
    df: pd.DataFrame, city: str, include_no_place: bool = False
) -> pd.DataFrame:
    """
    Compute annual counts for the target city vs other locations, with an optional
    series for records lacking a place of publication.
    """
    df = df.copy()
    mask = df["year_min"].between(YEAR_START, YEAR_END)
    df = df.loc[mask]
    if df.empty:
        raise ValueError(
            f"No rows between {YEAR_START} and {YEAR_END} for city='{city}'."
        )

    df["year_min"] = df["year_min"].astype(int)
    city_group = df["city_group"].fillna("")
    df["category"] = np.select(
        [
            city_group == city,
            city_group == "No place of publication",
        ],
        ["city", "no_place"],
        default="other",
    )

    grouped_full = (
        df.groupby(["year_min", "category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["city", "other", "no_place"], fill_value=0)
    )
    years = pd.Index(range(YEAR_START, YEAR_END + 1), name="year_min")
    grouped_full = grouped_full.reindex(years, fill_value=0)

    total_records = len(df)
    grouped_total = int(grouped_full.to_numpy().sum())
    if grouped_total != total_records:
        print(
            f"Warning: summed annual counts ({grouped_total}) differ from filtered rows ({total_records})."
        )

    columns = ["city", "other"]
    if include_no_place:
        columns.append("no_place")
    return grouped_full[columns]


def main():
    parser = argparse.ArgumentParser(
        description="Plot absolute PS imprint counts in New York vs other locations"
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
        help="City to highlight (default: New York City)",
    )
    parser.add_argument(
        "--output-line",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for line chart (default: figures/outputs/fig2.png)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Include 95 pct confidence intervals for the plotted series",
    )
    parser.add_argument(
        "--include-no-place",
        action="store_true",
        help='Plot "No place of publication" as its own series',
    )
    args = parser.parse_args()

    counts = compute_counts(
        load_data(args.input_csv),
        args.city,
        include_no_place=args.include_no_place,
    )
    years = counts.index.to_numpy()
    city_counts = counts["city"].to_numpy()
    other_counts = counts["other"].to_numpy()
    no_place_counts = counts["no_place"].to_numpy() if "no_place" in counts else None

    style.apply_style()
    plt.figure()
    if args.ci:
        z = 1.96
        se_city = np.sqrt(city_counts)
        lower_city = (city_counts - z * se_city).clip(lower=0)
        upper_city = city_counts + z * se_city
        plt.fill_between(
            years, lower_city, upper_city, color=style.COLOR_NYC, alpha=0.15
        )
        se_other = np.sqrt(other_counts)
        lower_other = (other_counts - z * se_other).clip(lower=0)
        upper_other = other_counts + z * se_other
        plt.fill_between(
            years, lower_other, upper_other, color=style.COLOR_OTHER, alpha=0.15
        )
        if no_place_counts is not None:
            se_no_place = np.sqrt(no_place_counts)
            lower_no_place = (no_place_counts - z * se_no_place).clip(lower=0)
            upper_no_place = no_place_counts + z * se_no_place
            plt.fill_between(
                years,
                lower_no_place,
                upper_no_place,
                color=style.COLOR_NOPLACE,
                alpha=0.15,
            )
    plt.plot(
        years,
        city_counts,
        label=args.city,
        color=style.COLOR_NYC,
        linestyle="-",
        marker=style.MARKERS[0],
        markevery=10,
        markersize=4,
    )
    plt.plot(
        years,
        other_counts,
        label="Other",
        color=style.COLOR_OTHER,
        linestyle="--",
        marker=style.MARKERS[1],
        markevery=10,
        markersize=4,
    )
    if no_place_counts is not None:
        plt.plot(
            years,
            no_place_counts,
            label="No place of publication",
            color=style.COLOR_NOPLACE,
            linestyle=":",
            marker=style.MARKERS[2],
            markevery=10,
            markersize=4,
        )
    plt.xlabel("Year")
    plt.ylabel("PS records with place of publication")
    plt.legend()
    plt.tight_layout()
    style.save_figure(args.output_line)


if __name__ == "__main__":
    main()
