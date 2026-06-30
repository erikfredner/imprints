#!/usr/bin/env python3
"""
Generate a line chart of the New-York-imprint share over time for the largest
PS numerical sub-ranges, applying the same NYC plotting rule as fig1 (NYC as a
share of placed records) to each range separately. Lines are labelled at the
right edge with their record counts instead of a legend.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style
from imprints.ps_ranges import FEATURED_KEYS, RANGE_LABELS
from range_shares import counts_matrices, despread_labels, share_matrix

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig5.png"
YEAR_START = 1900
YEAR_END = 2010

#: Grayscale-legible line styles cycled across the featured ranges.
LINE_STYLES = [
    {"linestyle": "-", "marker": "o"},
    {"linestyle": "--", "marker": "s"},
    {"linestyle": "-.", "marker": "^"},
    {"linestyle": ":", "marker": "D"},
    {"linestyle": "-", "marker": "v"},
]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot NYC imprint share over time for the largest PS sub-ranges"
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
        "--window",
        type=int,
        default=5,
        help="Rolling window size in years for smoothing (default: 5)",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply rolling smoothing to annual counts before the share (default: true)",
    )
    parser.add_argument(
        "--min-year-n",
        type=int,
        default=20,
        help="Drop a range's year cells with fewer placed records than this, so "
        "sparse early years don't spike the line (default: 20)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: figures/outputs/fig5.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    nyc, other = counts_matrices(
        df, YEAR_START, YEAR_END, city=args.city, ranges=FEATURED_KEYS
    )
    share = share_matrix(
        nyc, other, window=args.window, smooth=args.smooth, min_n=args.min_year_n
    )
    totals = (nyc + other).sum()  # raw N per range, unsmoothed

    style.apply_style()
    fig, ax = plt.subplots()

    endpoints = []  # (last_y, label) for right-edge annotation
    for i, key in enumerate(FEATURED_KEYS):
        series = share[key].dropna()
        if series.empty:
            continue
        line_style = LINE_STYLES[i % len(LINE_STYLES)]
        ax.plot(
            series.index,
            series.values,
            color="black",
            markevery=10,
            markersize=4,
            linewidth=1.2,
            **line_style,
        )
        n = int(round(totals[key]))
        label = f"{RANGE_LABELS[key]}  (N={n:,})"
        endpoints.append([float(series.iloc[-1]), label])

    ax.axhline(50, color="gray", linestyle="dotted", linewidth=1)
    ax.set_xlim(YEAR_START, YEAR_END)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of placed PS works published in " + args.city)

    # Right-edge direct labels in place of a legend.
    y0, y1 = ax.get_ylim()
    min_gap = 0.06 * (y1 - y0)
    raw_y = [e[0] for e in endpoints]
    label_y = despread_labels(raw_y, min_gap)
    for (_, label), y in zip(endpoints, label_y):
        ax.annotate(
            label,
            xy=(YEAR_END, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=7,
        )

    fig.subplots_adjust(right=0.6)
    style.save_figure(args.output)

    # Per-range summary stats, echoing fig1's reporting.
    for key in FEATURED_KEYS:
        series = share[key].dropna()
        if series.empty:
            print(f"{key}: no data")
            continue
        peak_year = int(series.idxmax())
        print(
            f"{key} ({RANGE_LABELS[key]}): "
            f"start {series.index.min()}={series.iloc[0]:.1f}%, "
            f"peak {peak_year}={series.max():.1f}%, "
            f"end {series.index.max()}={series.iloc[-1]:.1f}%, "
            f"N={int(round(totals[key])):,}"
        )


if __name__ == "__main__":
    main()
