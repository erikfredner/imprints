#!/usr/bin/env python3
"""
Generate a single-panel map of the continental US showing, for each
publication-location coordinate, whether Library of Congress class PS
(American literary) works published there skewed toward the period before or
after New York City's peak share year.

Unlike nyc_peak_map.py's two-panel before/after split, every coordinate stays
on one map here, and the before/after comparison is encoded as color: a
diverging scale from blue (published disproportionately more before the
peak) through gray (roughly even) to vermillion (disproportionately more
after). "Disproportionately" is relative to each period's own total volume --
a coordinate's *share* of all US/CONUS PS records in that period, not its raw
record count -- since the post-peak period has far more total records
overall and raw counts would make nearly every point look like it grew.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

import nyc_peak_map as npm
import style

DEFAULT_INPUT = npm.DEFAULT_INPUT
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "outputs/nyc_peak_map_diverging.png"
)
YEAR_START = npm.YEAR_START
YEAR_END = npm.YEAR_END

#: Diverging colormap: pre-peak-dominant pole, neutral midpoint, post-peak-
#: dominant pole. Reuses the Okabe-Ito blue/vermillion already used elsewhere
#: in this figure family, so "pre-peak" and "post-peak" read as opposite
#: poles the way NYC/Other do in nyc_peak_map.py, with a hue-free gray at the
#: neutral midpoint (never a hue there -- see style guidance).
PRE_PEAK_COLOR = style.OKABE_ITO[0]  # blue
POST_PEAK_COLOR = style.OKABE_ITO[1]  # vermillion
NEUTRAL_COLOR = "0.85"
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "pre_post_peak", [PRE_PEAK_COLOR, NEUTRAL_COLOR, POST_PEAK_COLOR]
)
DIVERGING_NORM = Normalize(vmin=-1, vmax=1)

MIN_MARKER_SIZE = npm.MIN_MARKER_SIZE
MAX_MARKER_SIZE = npm.MAX_MARKER_SIZE
POINT_ALPHA = npm.POINT_ALPHA
LEGEND_COUNTS = npm.LEGEND_COUNTS


def coordinate_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count records at each unique (lat, lon) coordinate."""
    return (
        df.groupby(["llm_nominatim_lat", "llm_nominatim_lon"])
        .size()
        .rename("count")
        .reset_index()
    )


def compute_share_diff(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """For every coordinate appearing in `before` and/or `after`, compute its
    share of each period's total US/CONUS PS records and a bounded diverging
    metric of the difference between the two shares.

    metric = (share_after - share_before) / (share_after + share_before),
    which ranges from -1 (the coordinate published only before the peak) to
    +1 (only after), through 0 (an equal share of both periods' output). A
    share, not a raw count, is compared: the after-peak period has many more
    total records, so comparing raw counts would make nearly every
    coordinate look like it grew.
    """
    before_counts = coordinate_counts(before).rename(columns={"count": "count_before"})
    after_counts = coordinate_counts(after).rename(columns={"count": "count_after"})
    merged = before_counts.merge(
        after_counts,
        on=["llm_nominatim_lat", "llm_nominatim_lon"],
        how="outer",
    ).fillna({"count_before": 0, "count_after": 0})

    share_before = merged["count_before"] / len(before)
    share_after = merged["count_after"] / len(after)

    merged["share_before"] = share_before
    merged["share_after"] = share_after
    merged["metric"] = (share_after - share_before) / (share_after + share_before)
    merged["total_count"] = merged["count_before"] + merged["count_after"]
    return merged


def plot_map(coords: pd.DataFrame, peak_year: int, output_path: Path) -> None:
    style.apply_style()
    fig, ax = plt.subplots(
        figsize=(9, 6.5), subplot_kw={"projection": npm.make_projection()}
    )

    npm.add_basemap(ax)
    counts = coords["total_count"].to_numpy()
    sizes = npm.marker_size(counts, counts)
    mappable = ax.scatter(
        coords["llm_nominatim_lon"],
        coords["llm_nominatim_lat"],
        s=sizes,
        c=coords["metric"],
        cmap=DIVERGING_CMAP,
        norm=DIVERGING_NORM,
        alpha=POINT_ALPHA,
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )
    fig.suptitle(
        "Publication locations of Library of Congress class PS "
        "(American literature) records"
    )
    ax.set_title(
        f"Color: share of output before vs. after {peak_year} (NYC's peak share year)",
        fontsize=10,
    )

    cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", fraction=0.035, pad=0.02)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(
        [f"Pre-{peak_year}\ndominant", "Equal\nshare", f"Post-{peak_year}\ndominant"]
    )
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(top=0.88, bottom=0.14)
    npm.add_size_legend(fig, counts)
    style.save_figure(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PS publication locations on a single map, colored by "
        "whether each location's share of output was greater before or after "
        "NYC's peak share year"
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
        "(default: figures/outputs/nyc_peak_map_diverging.png)",
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

    coords = compute_share_diff(before, after)
    only_pre = (coords["metric"] == -1).sum()
    only_post = (coords["metric"] == 1).sum()

    print(f"NYC peak share year: {peak_year}")
    print(
        f"{len(coords):,} unique coordinates across both periods "
        f"({len(before):,} pre-peak records, {len(after):,} post-peak records)"
    )
    print(
        f"{only_pre:,} coordinates published only through {peak_year}, "
        f"{only_post:,} published only after {peak_year}"
    )

    plot_map(coords, peak_year, args.output)


if __name__ == "__main__":
    main()
