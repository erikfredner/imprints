#!/usr/bin/env python3
"""
Show how the make-up of PS changes over time, broken down by numerical
sub-range, as two stacked-area charts: absolute record counts (fig6_counts) and
the same stack normalized to 100% per year (fig6_share). Together they show both
the growth in PS volume and the shifting balance among ranges (e.g. the rise of
the later "individual authors" ranges).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from imprints.ps_ranges import RANGE_LABELS, RANGE_ORDER, add_range_column

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig6.png"
YEAR_START = 1900
YEAR_END = 2010

#: Hatches cycled across bands to keep adjacent grays distinguishable.
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "+++", "ooo"]


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def compute_range_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Year x range matrix of record counts over the full PS partition.

    Includes every placed and unplaced PS record (this view is about range
    volume, not NYC share); rows outside every range are dropped.
    """
    df = add_range_column(df)
    df = df[df["range"].notna()]
    df = df[df["year_min"].between(YEAR_START, YEAR_END)].copy()
    df["year_min"] = df["year_min"].astype(int)
    years = pd.Index(range(YEAR_START, YEAR_END + 1), name="year_min")
    matrix = (
        df.groupby(["year_min", "range"])
        .size()
        .unstack("range")
        .reindex(index=years, columns=RANGE_ORDER)
        .fillna(0.0)
    )
    return matrix


def _output_paths(output: Path) -> tuple[Path, Path]:
    """Derive the two real output base paths from ``--output``."""
    return (
        output.with_name(f"{output.stem}_counts{output.suffix}"),
        output.with_name(f"{output.stem}_share{output.suffix}"),
    )


def _plot_stack(matrix: pd.DataFrame, ylabel: str, output: Path) -> None:
    """Draw one stacked-area chart with right-edge band labels and save it."""
    style.apply_style()
    fig, ax = plt.subplots()

    years = matrix.index.to_numpy()
    colors = [
        style.OKABE_ITO[i % len(style.OKABE_ITO)] for i in range(len(RANGE_ORDER))
    ]
    stacks = ax.stackplot(
        years,
        [matrix[key].to_numpy() for key in RANGE_ORDER],
        colors=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    for poly, hatch in zip(stacks, HATCHES):
        poly.set_hatch(hatch)

    # Right-margin labels, evenly spaced top-to-bottom and joined to each band's
    # right-edge centre by a thin leader, so the thin bottom bands stay legible.
    final = matrix.iloc[-1]
    centers = (final.cumsum() - final / 2).reindex(RANGE_ORDER)
    keys = [key for key in reversed(RANGE_ORDER) if final[key] > 0]
    if keys:
        top = float(matrix.sum(axis=1).max())
        label_ys = np.linspace(0.96 * top, 0.04 * top, len(keys))
        for key, y in zip(keys, label_ys):
            ax.annotate(
                RANGE_LABELS[key],
                xy=(YEAR_END, centers[key]),
                xytext=(YEAR_END + 2, y),
                textcoords="data",
                va="center",
                ha="left",
                fontsize=6,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-", lw=0.4, color=style.COLOR_REFERENCE),
            )

    ax.set_xlim(YEAR_START, YEAR_END)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(right=0.62)
    style.save_figure(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Stacked-area composition of PS by numerical sub-range over time"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Base output path; _counts and _share variants are derived from it",
    )
    args = parser.parse_args()

    matrix = compute_range_matrix(load_data(args.input_csv))
    counts_path, share_path = _output_paths(args.output)

    _plot_stack(matrix, "PS records", counts_path)

    totals = matrix.sum(axis=1).replace(0, np.nan)
    share = matrix.div(totals, axis=0) * 100
    _plot_stack(share, "% of PS records", share_path)


if __name__ == "__main__":
    main()
