#!/usr/bin/env python3
"""
Generate a line chart of the New-York-imprint share over time for the largest
PS numerical sub-ranges, applying the same NYC plotting rule as ps_nyc_share (NYC as a
share of placed records) to each range separately. Ranges with too few records
to plot reliably are dropped; the remainder are ordered by PS range (not by
record count) and identified in a legend below the plotting area.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style
from imprints.ps_ranges import RANGE_LABELS, RANGE_ORDER
from range_shares import counts_matrices, share_matrix

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/ps_range_nyc_share.png"
YEAR_START = 1900
YEAR_END = 2010
YEAR_MARGIN = 2
MIN_RANGE_N = 10_000
LEGEND_HEIGHT_INCHES = 1.1


def format_count(n: int, compact: bool = False) -> str:
    """Format a legend count, abbreviating thousands when space is tight."""
    if compact and n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return f"{n:,}"


def range_label(key: str, n: int, compact_count: bool = False) -> str:
    """Return a range-prefixed legend label."""
    return f"{key}  {RANGE_LABELS[key]}  (N={format_count(n, compact_count)})"


def add_bottom_legend(fig, ax, handles, keys, totals):
    """Add the range legend below the axes, compacting counts if necessary."""

    anchor_display = ax.transAxes.transform((0.5, -0.18))
    anchor_y = fig.transFigure.inverted().transform(anchor_display)[1]

    def make_legend(
        compact_count: bool,
        fontsize: float = 7,
        handlelength: float = 2.5,
        columnspacing: float = 1.5,
    ):
        labels = [
            range_label(key, int(round(totals[key])), compact_count) for key in keys
        ]
        return fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, anchor_y),
            bbox_transform=fig.transFigure,
            ncol=2,
            frameon=False,
            fontsize=fontsize,
            handlelength=handlelength,
            columnspacing=columnspacing,
        )

    legend = make_legend(compact_count=False)
    fig.canvas.draw()
    available_width = fig.get_window_extent().width - fig.dpi * 0.2
    if legend.get_window_extent().width > available_width:
        legend.remove()
        legend = make_legend(compact_count=True)
        fig.canvas.draw()
    if legend.get_window_extent().width > available_width:
        legend.remove()
        legend = make_legend(
            compact_count=True,
            fontsize=6,
            handlelength=2,
            columnspacing=0.8,
        )
    return legend


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
        default=5,
        help="Drop a range's year cells with fewer placed records than this, so "
        "sparse early years don't spike the line (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: "
        "figures/outputs/ps_range_nyc_share.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    nyc, other = counts_matrices(
        df, YEAR_START, YEAR_END, city=args.city, ranges=RANGE_ORDER
    )
    share = share_matrix(
        nyc, other, window=args.window, smooth=args.smooth, min_n=args.min_year_n
    )
    totals = (nyc + other).sum()  # raw N per range, unsmoothed
    # RANGE_ORDER is already ascending by PS number; filtering (not sorting)
    # keeps that order so the plot and legend read low-to-high PS range.
    featured_keys = [key for key in RANGE_ORDER if totals[key] >= MIN_RANGE_N]

    style.apply_style()
    base_width, base_height = plt.rcParams["figure.figsize"]
    figure_height = base_height + LEGEND_HEIGHT_INCHES
    fig, ax = plt.subplots(figsize=(base_width, figure_height))

    handles = []
    plotted_keys = []
    for i, key in enumerate(featured_keys):
        series = share[key].dropna()
        if series.empty:
            continue
        (line,) = ax.plot(
            series.index,
            series.values,
            markevery=10,
            markersize=4,
            linewidth=1.2,
            **style.series_style(i),
        )
        handles.append(line)
        plotted_keys.append(key)

    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)
    ax.set_xlim(YEAR_START - YEAR_MARGIN, YEAR_END + YEAR_MARGIN)
    ax.set_xlabel("Year")
    ax.set_ylabel("PS share with NYC imprint")
    style.percent_yaxis(ax)

    # Give the axes the same physical layout area as ps_nyc_share's default 6.4 x 4.8
    # figure, reserving only the added height for this figure's bottom legend.
    fig.tight_layout(rect=(0, LEGEND_HEIGHT_INCHES / figure_height, 1, 1))
    add_bottom_legend(fig, ax, handles, plotted_keys, totals)
    style.save_figure(args.output)

    # Per-range summary stats, echoing ps_nyc_share's reporting.
    for key in featured_keys:
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
