#!/usr/bin/env python3
"""
Generate a two-panel map of the continental US showing where Library of
Congress class PS (American literary) works were published, before vs. after
New York City's peak share year.

The simplest possible version of this story: just two color/marker
categories (New York City, Other), point size = raw record count on a single
scale shared across both panels. NYC's share of PS output falls after its
peak year not because NYC's own output shrinks but because output elsewhere
grows -- so with a shared size scale, that reads directly off the map: NYC's
bubble stays about the same size in both panels, while the bubbles
everywhere else multiply and grow in the after panel. Unlike nyc_peak_map.py,
there is no third "new location" category -- whether an Other coordinate
already published before the peak isn't the point being illustrated, and
dropping it removes a legend distinction that doesn't serve the story.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import nyc_peak_map as npm
import style

DEFAULT_INPUT = npm.DEFAULT_INPUT
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/nyc_peak_map_simple.png"
YEAR_START = npm.YEAR_START
YEAR_END = npm.YEAR_END


def plot_map(
    before_nyc,
    before_other,
    after_nyc,
    after_other,
    all_counts: np.ndarray,
    start_year: int,
    peak_year: int,
    end_year: int,
    n_before: int,
    n_after: int,
    output_path: Path,
) -> None:
    style.apply_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 9.5), subplot_kw={"projection": npm.make_projection()}
    )

    npm.plot_panel(
        axes[0],
        [
            (before_nyc, npm.NYC_COLOR, npm.NYC_MARKER),
            (before_other, npm.OTHER_COLOR, npm.OTHER_MARKER),
        ],
        all_counts,
        f"{start_year}-{peak_year} (NYC peak) ({n_before:,} records)",
    )
    npm.plot_panel(
        axes[1],
        [
            (after_nyc, npm.NYC_COLOR, npm.NYC_MARKER),
            (after_other, npm.OTHER_COLOR, npm.OTHER_MARKER),
        ],
        all_counts,
        f"{peak_year}-{end_year} ({n_after:,} records)",
    )
    npm.add_category_legend(
        axes[0],
        [
            ("New York City", npm.NYC_COLOR, npm.NYC_MARKER),
            ("Other", npm.OTHER_COLOR, npm.OTHER_MARKER),
        ],
    )

    fig.subplots_adjust(top=0.95, bottom=0.09, hspace=0.25)
    npm.add_size_legend(fig, all_counts)
    style.save_figure(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PS publication locations before vs. after NYC's peak "
        "share year, using only NYC/Other categories and a shared raw-count size "
        "scale (the simplest version of this figure)"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to joined PS/Nominatim data CSV (default: data/PS/geocoded.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure "
        "(default: figures/outputs/nyc_peak_map_simple.png)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=YEAR_START,
        help="Start year (inclusive, default: 1900)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=YEAR_END,
        help="End year (inclusive, default: 2010)",
    )
    args = parser.parse_args()

    df = npm.load_data(args.input_csv)
    peak_year = npm.compute_peak_year(df, args.start_year, args.end_year)

    us = npm.filter_us_conus(df)
    us = us[us["year_min"].between(args.start_year, args.end_year)]
    before = us[us["year_min"] <= peak_year]
    after = us[us["year_min"] > peak_year]

    before_counts = npm.coordinate_counts(before)
    after_counts = npm.coordinate_counts(after)
    before_nyc, before_other = npm.split_by_city(before_counts)
    after_nyc, after_other = npm.split_by_city(after_counts)
    all_counts = np.concatenate(
        [before_counts["count"].to_numpy(), after_counts["count"].to_numpy()]
    )

    print(f"NYC peak share year: {peak_year}")
    print(
        f"Through {peak_year}: {len(before):,} US/CONUS records at "
        f"{len(before_counts):,} unique coordinates "
        f"({len(before_nyc):,} NYC, {len(before_other):,} Other)"
    )
    print(
        f"After {peak_year}: {len(after):,} US/CONUS records at "
        f"{len(after_counts):,} unique coordinates "
        f"({len(after_nyc):,} NYC, {len(after_other):,} Other)"
    )

    plot_map(
        before_nyc,
        before_other,
        after_nyc,
        after_other,
        all_counts,
        args.start_year,
        peak_year,
        args.end_year,
        len(before),
        len(after),
        args.output,
    )


if __name__ == "__main__":
    main()
