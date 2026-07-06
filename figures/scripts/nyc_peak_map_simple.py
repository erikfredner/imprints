#!/usr/bin/env python3
"""
Generate a two-panel map of the continental US showing where Library of
Congress class PS (American literary) works were published, before vs. after
New York City's peak share year.

Color (not NYC/Other category) carries the record count at each coordinate,
on a shared log-scaled viridis colorbar -- useful for spotting the single
highest-volume coordinates in each period regardless of city. Point size is
also on a single scale shared across both panels: NYC's share of PS output
falls after its peak year not because NYC's own output shrinks but because
output elsewhere grows -- so with a shared size scale, that reads directly
off the map.
"""

import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

import ps_nyc_share
import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/geocoded.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/nyc_peak_map_simple.png"
YEAR_START = 1900
YEAR_END = 2010

#: Continental US bounding box. Excludes Alaska, Hawaii, Puerto Rico, and
#: other territories, which geocoded_country_code == "us" alone does not.
CONUS_LON_MIN, CONUS_LON_MAX = -125, -66
CONUS_LAT_MIN, CONUS_LAT_MAX = 24, 50

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
    itself reports, reused rather than reimplemented. Runs on the full
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
        title="PS records published at coordinates",
        loc="lower center",
        ncol=len(counts_in_range),
        bbox_to_anchor=(0.5, 0.0),
        frameon=False,
    )


def plot_viridis_map(
    before_counts,
    after_counts,
    all_counts: np.ndarray,
    start_year: int,
    peak_year: int,
    end_year: int,
    output_path: Path,
) -> None:
    """Color -- not NYC/Other category -- carries the record count at each
    coordinate, on one shared log-scaled viridis scale (log because counts
    span orders of magnitude). Point size also scales with count, on the
    same shared scale across both panels, so low-count points stay visible
    rather than shrinking to a single dark pixel."""
    style.apply_style()
    fig, axes = plt.subplots(
        2, 1, figsize=(8, 9.5), subplot_kw={"projection": make_projection()}
    )
    norm = LogNorm(vmin=all_counts.min(), vmax=all_counts.max())

    panels = [
        (axes[0], before_counts, f"{start_year}-{peak_year} (NYC peak)"),
        (axes[1], after_counts, f"{peak_year}-{end_year}"),
    ]
    mappable = None
    for ax, counts, title in panels:
        add_basemap(ax)
        # Sort ascending by count so, within this single scatter call, the
        # highest-count coordinates draw on top rather than whichever
        # category happens to plot last.
        counts = counts.sort_values("count")
        sizes = marker_size(counts["count"].to_numpy(), all_counts)
        mappable = ax.scatter(
            counts["geocoded_lon"],
            counts["geocoded_lat"],
            s=sizes,
            c=counts["count"],
            cmap="viridis",
            norm=norm,
            alpha=POINT_ALPHA,
            edgecolors="none",
            transform=ccrs.PlateCarree(),
            zorder=3,
        )
        ax.set_title(title)

    fig.subplots_adjust(top=0.95, bottom=0.09, hspace=0.25, right=0.86)
    cbar = fig.colorbar(
        mappable, ax=axes.tolist(), orientation="vertical", fraction=0.035, pad=0.02
    )
    cbar.set_label("PS records published at coordinates")
    add_size_legend(fig, all_counts)
    style.save_figure(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot PS publication locations before vs. after NYC's peak "
        "share year, with color and size carrying the record count at each "
        "coordinate on a shared scale"
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

    plot_viridis_map(
        before_counts,
        after_counts,
        all_counts,
        args.start_year,
        peak_year,
        args.end_year,
        args.output,
    )


if __name__ == "__main__":
    main()
