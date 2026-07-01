#!/usr/bin/env python3
"""
Variant of fig1.py: plots NYC share of PS-class imprints twice -- once for
all records, once with is_secondary == True (criticism/scholarship/reference,
per imprints.secondary_classification) excluded, leaving primary literature
only.
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

import fig1
import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_SECONDARY_CSV = (
    Path(__file__).resolve().parents[2] / "data/PS/secondary_classification.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig1_primary_only.png"

_LCCN_DIGITS_RE = re.compile(r"\d+")


def normalize_lccn(value):
    """Extract the leading run of digits from an lccn value as an int.

    data.csv's lccn column mixes plain-numeric and MARC-style values with
    trailing revision suffixes (e.g. "064012140 //r97"), which makes a raw
    string/int merge against secondary_classification.csv silently drop
    matches. Both sides are normalized through this before joining.
    """
    if pd.isna(value):
        return None
    m = _LCCN_DIGITS_RE.search(str(value))
    return int(m.group()) if m else None


def merge_secondary(df: pd.DataFrame, secondary_csv: Path) -> pd.DataFrame:
    """Attach is_secondary to df by joining on a normalized lccn key."""
    sec = pd.read_csv(secondary_csv, usecols=["lccn", "is_secondary"])
    df = df.copy()
    df["lccn_norm"] = df["lccn"].map(normalize_lccn)
    sec["lccn_norm"] = sec["lccn"].map(normalize_lccn)
    sec = sec.dropna(subset=["lccn_norm"])

    merged = df.merge(sec[["lccn_norm", "is_secondary"]], on="lccn_norm", how="left")
    unmatched = merged["is_secondary"].isna().sum()
    print(
        f"Rows without a secondary-classification match: {unmatched:,} "
        f"({unmatched / len(merged):.2%}); kept in the primary-only line."
    )
    return merged


def plot_comparison(
    pct_all: pd.DataFrame,
    pct_primary: pd.DataFrame,
    city: str,
    output_path: str = None,
) -> None:
    """Plot both NYC-share lines (all records vs. primary-literature-only)."""
    style.apply_style()
    fig, ax = plt.subplots()

    ax.plot(
        pct_all.index.values,
        pct_all[1],
        color=style.COLOR_NYC,
        linestyle=style.LINESTYLES[0],
        label="All records",
    )
    ax.plot(
        pct_primary.index.values,
        pct_primary[1],
        color=style.COLOR_NYC,
        linestyle=style.LINESTYLES[1],
        label="Primary literature only (secondary excluded)",
    )
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of PS works published in " + city)
    ax.legend()
    style.percent_yaxis(ax)
    plt.tight_layout()
    if output_path:
        style.save_figure(output_path)
    else:
        plt.show()


def print_stats(label: str, pct_city: pd.Series) -> None:
    pct_rounded = pct_city.round()
    at_least_50 = pct_rounded[pct_rounded >= 50]
    year_first_at_least_50 = (
        int(at_least_50.index.min()) if not at_least_50.empty else None
    )
    year_last_at_least_50 = (
        int(at_least_50.index.max()) if not at_least_50.empty else None
    )
    year_peak = int(pct_city.idxmax()) if not pct_city.empty else None
    pct_peak = float(pct_city.loc[year_peak]) if year_peak is not None else None

    print(f"=== {label} ===")
    print("First year NYC share >= 50% (rounded):", year_first_at_least_50)
    print(
        "Peak year for NYC:",
        year_peak,
        f"({round(pct_peak, 2)}%)" if pct_peak is not None else "",
    )
    print("Last year NYC share >= 50% (rounded):", year_last_at_least_50)
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot PS imprints published in New York vs. other locations, "
            "comparing all records to primary-literature-only (secondary "
            "excluded)."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--secondary-csv",
        type=Path,
        default=DEFAULT_SECONDARY_CSV,
        help=(
            "Path to secondary-classification CSV (default: "
            "data/PS/secondary_classification.csv)"
        ),
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
        help="Apply rolling smoothing to annual percentages (default: true)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: figures/outputs/fig1_primary_only.png)",
    )
    args = parser.parse_args()

    city = "New York City" if args.city == "New York" else args.city

    df = fig1.load_data(args.input_csv)
    df = merge_secondary(df, args.secondary_csv)
    df_primary = df[df["is_secondary"] != True]  # noqa: E712 (keeps False and NaN)

    pct_all = fig1.compute_city_share(
        df,
        city=city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        smooth=args.smooth,
    )
    pct_primary = fig1.compute_city_share(
        df_primary,
        city=city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        smooth=args.smooth,
    )

    print_stats("All records", pct_all[1])
    print_stats("Primary literature only (secondary excluded)", pct_primary[1])

    plot_comparison(pct_all, pct_primary, city=city, output_path=args.output)


if __name__ == "__main__":
    main()
