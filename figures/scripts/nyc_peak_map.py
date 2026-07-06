#!/usr/bin/env python3
"""
Generate a two-panel map of the continental US showing where Library of
Congress class PS (American literary) works were published, before vs. after
New York City's peak share year.

Point size is on a single scale shared across both panels, so the greater
volume of publication locations after NYC's peak is visible relative to the
concentrated pre-peak panel. Points use one of three color/shape categories:
New York City, Other (a non-NYC coordinate that already published PS works
before the peak), and New (a non-NYC coordinate appearing only after the
peak, shown only in the lower panel).
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import ps_nyc_share
import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/geocoded.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/nyc_peak_map.png"
YEAR_START = 1900
YEAR_END = 2010

#: Continental US bounding box. Excludes Alaska, Hawaii, Puerto Rico, and
#: other territories, which geocoded_country_code == "us" alone does not.
CONUS_LON_MIN, CONUS_LON_MAX = -125, -66
CONUS_LAT_MIN, CONUS_LAT_MAX = 24, 50

#: Coordinates whose city_group is "New York City", in either panel.
NYC_COLOR = style.COLOR_NYC
NYC_MARKER = "o"
#: Non-NYC coordinates that already published PS works before the peak year.
OTHER_COLOR = style.COLOR_OTHER
OTHER_MARKER = "s"
#: Non-NYC coordinates appearing only after the peak year (post-peak panel only).
NEW_COLOR = style.OKABE_ITO[2]
NEW_MARKER = "^"
POINT_ALPHA = 0.5
MIN_MARKER_SIZE = 8
MAX_MARKER_SIZE = 600
#: Reference counts shown in the shared-scale size legend.
LEGEND_COUNTS = [1, 10, 100, 1000]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load the joined PS/geocoded data (see imprints.join_geocoded)."""
    return pd.read_csv(csv_path)


def filter_us_conus(df: pd.DataFrame) -> pd.DataFrame:
    """Restrict to US-resolved records within the continental bounding box."""
    us = df[
        (df["geocoded_country_code"] == "us")
        & df["geocoded_lat"].notna()
        & df["geocoded_lon"].notna()
    ]
    return us[
        us["geocoded_lon"].between(CONUS_LON_MIN, CONUS_LON_MAX)
        & us["geocoded_lat"].between(CONUS_LAT_MIN, CONUS_LAT_MAX)
    ]


def compute_peak_year(df: pd.DataFrame, start_year: int, end_year: int) -> int:
    """NYC's peak share year over [start_year, end_year], via
    ps_nyc_share.compute_city_share -- the same computation ps_nyc_share.py
    itself reports, reused rather than reimplemented (see
    new_publisher_nyc_share.py for the same pattern). Runs on the full
    dataframe, not the US/CONUS subset: NYC's share is of all PS records,
    independent of geocoding success."""
    pct = ps_nyc_share.compute_city_share(
        df, city="New York City", start_year=start_year, end_year=end_year, smooth=False
    )
    return int(pct[1].idxmax())


def coordinate_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Count records at each unique (lat, lon) in df, flagging coordinates
    where any record's city_group is "New York City"."""
    return (
        df.groupby(["geocoded_lat", "geocoded_lon"])
        .agg(
            count=("city_group", "size"),
            is_nyc=("city_group", lambda s: (s == "New York City").any()),
        )
        .reset_index()
    )


def split_by_city(coord_counts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split coordinate counts into NYC and non-NYC (Other) groups."""
    nyc = coord_counts[coord_counts["is_nyc"]]
    other = coord_counts[~coord_counts["is_nyc"]]
    return nyc, other


def split_by_prior_presence(
    after_counts: pd.DataFrame, before_counts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split after_counts into coordinates that also appear in before_counts
    (returning locations) and those that don't (new locations)."""
    merged = after_counts.merge(
        before_counts[["geocoded_lat", "geocoded_lon"]],
        on=["geocoded_lat", "geocoded_lon"],
        how="left",
        indicator=True,
    )
    existing = merged[merged["_merge"] == "both"].drop(columns="_merge")
    new = merged[merged["_merge"] == "left_only"].drop(columns="_merge")
    return existing, new


def marker_size(counts, all_counts: np.ndarray) -> np.ndarray:
    """Map counts to marker size on one scale derived from all_counts (the
    union of both panels' counts), so a panel with larger counts renders
    visibly larger markers rather than each panel auto-scaling to its own
    max."""
    denom = np.sqrt(all_counts.max())
    return (
        MIN_MARKER_SIZE + (MAX_MARKER_SIZE - MIN_MARKER_SIZE) * np.sqrt(counts) / denom
    )


def make_projection() -> ccrs.Projection:
    """Equal-area projection centered on the continental US, so panel-to-panel
    density comparisons aren't skewed by latitude-dependent area distortion."""
    return ccrs.AlbersEqualArea(
        central_longitude=-96,
        central_latitude=37.5,
        standard_parallels=(29.5, 45.5),
    )


def add_basemap(ax) -> None:
    """Add US coastline/state/border features and set the CONUS extent."""
    ax.set_extent(
        [CONUS_LON_MIN, CONUS_LON_MAX, CONUS_LAT_MIN, CONUS_LAT_MAX],
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor="0.6")
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)


def scatter_coords(
    ax, coord_counts: pd.DataFrame, all_counts: np.ndarray, color: str, marker: str
) -> None:
    """Plot one color/marker group of sized, alpha'd points at their
    coordinates."""
    sizes = marker_size(coord_counts["count"].to_numpy(), all_counts)
    ax.scatter(
        coord_counts["geocoded_lon"],
        coord_counts["geocoded_lat"],
        s=sizes,
        c=color,
        marker=marker,
        alpha=POINT_ALPHA,
        edgecolors="none",
        transform=ccrs.PlateCarree(),
        zorder=3,
    )


def plot_panel(
    ax,
    groups: list[tuple[pd.DataFrame, str, str]],
    all_counts: np.ndarray,
    title: str,
) -> None:
    """Draw one scatter-map panel: US basemap features plus one or more
    color/marker groups of sized, alpha'd points."""
    add_basemap(ax)
    for coord_counts, color, marker in groups:
        scatter_coords(ax, coord_counts, all_counts, color, marker)
    ax.set_title(title)


def add_size_legend(fig, all_counts: np.ndarray) -> None:
    """Reference bubbles at fixed counts, sized with the same shared scale as
    the panels, so readers can translate marker size back to a record count."""
    counts_in_range = [c for c in LEGEND_COUNTS if c <= all_counts.max()]
    handles = [
        plt.scatter(
            [],
            [],
            s=marker_size(np.array([c]), all_counts),
            c="0.3",
            alpha=POINT_ALPHA,
            edgecolors="none",
        )
        for c in counts_in_range
    ]
    fig.legend(
        handles,
        [f"{c:,}" for c in counts_in_range],
        title="Records at coordinate",
        loc="lower center",
        ncol=len(counts_in_range),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )


def add_category_legend(ax, entries: list[tuple[str, str, str]]) -> None:
    """Legend mapping each (label, color, marker) entry to a proxy point,
    so panels can show only the categories they actually contain."""
    handles = [
        Line2D(
            [],
            [],
            marker=marker,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor="none",
            markersize=9 if marker == NEW_MARKER else 8,
        )
        for _, color, marker in entries
    ]
    labels = [label for label, _, _ in entries]
    ax.legend(
        handles,
        labels,
        loc="lower left",
        fontsize=8,
        framealpha=0.9,
        borderpad=0.6,
    )


def plot_map(
    before: pd.DataFrame, after: pd.DataFrame, peak_year: int, output_path: Path
) -> None:
    style.apply_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 9.5), subplot_kw={"projection": make_projection()}
    )

    before_counts = coordinate_counts(before)
    after_counts = coordinate_counts(after)
    before_nyc, before_other = split_by_city(before_counts)
    after_nyc, after_other = split_by_city(after_counts)
    after_other_existing, after_other_new = split_by_prior_presence(
        after_other, before_other
    )
    all_counts = np.concatenate(
        [before_counts["count"].to_numpy(), after_counts["count"].to_numpy()]
    )

    fig.suptitle(
        "Publication locations of Library of Congress class PS "
        "(American literature) records"
    )
    plot_panel(
        axes[0],
        [
            (before_nyc, NYC_COLOR, NYC_MARKER),
            (before_other, OTHER_COLOR, OTHER_MARKER),
        ],
        all_counts,
        f"Through {peak_year} (NYC's peak share year)",
    )
    plot_panel(
        axes[1],
        [
            (after_nyc, NYC_COLOR, NYC_MARKER),
            (after_other_existing, OTHER_COLOR, OTHER_MARKER),
            (after_other_new, NEW_COLOR, NEW_MARKER),
        ],
        all_counts,
        f"After {peak_year}",
    )
    add_category_legend(
        axes[0],
        [
            ("New York City", NYC_COLOR, NYC_MARKER),
            ("Other", OTHER_COLOR, OTHER_MARKER),
        ],
    )
    add_category_legend(
        axes[1],
        [
            ("New York City", NYC_COLOR, NYC_MARKER),
            (
                f"Other (also published here through {peak_year})",
                OTHER_COLOR,
                OTHER_MARKER,
            ),
            (f"New location after {peak_year}", NEW_COLOR, NEW_MARKER),
        ],
    )

    fig.subplots_adjust(top=0.93, bottom=0.09, hspace=0.2)
    add_size_legend(fig, all_counts)
    style.save_figure(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PS publication locations before vs. after NYC's peak share year"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to joined PS/geocoded data CSV (default: data/PS/geocoded.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: figures/outputs/nyc_peak_map.png)",
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

    df = load_data(args.input_csv)
    peak_year = compute_peak_year(df, args.start_year, args.end_year)

    us = filter_us_conus(df)
    us = us[us["year_min"].between(args.start_year, args.end_year)]
    before = us[us["year_min"] <= peak_year]
    after = us[us["year_min"] > peak_year]

    before_counts = coordinate_counts(before)
    after_counts = coordinate_counts(after)
    before_nyc, before_other = split_by_city(before_counts)
    after_nyc, after_other = split_by_city(after_counts)
    after_other_existing, after_other_new = split_by_prior_presence(
        after_other, before_other
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
        f"({len(after_nyc):,} NYC, {len(after_other_existing):,} Other "
        f"also active before the peak, {len(after_other_new):,} new)"
    )

    plot_map(before, after, peak_year, args.output)


if __name__ == "__main__":
    main()
