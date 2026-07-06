#!/usr/bin/env python3
"""
Two-panel figure combining ps_nyc_counts (NYC vs other PS imprint counts) on
the left with ps_unique_publishers (unique PS publisher counts per year) on
the right.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import ps_nyc_counts
import ps_unique_publishers
import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/nyc_and_publishers.png"


def plot_left(ax, args, counts) -> None:
    """Plot the ps_nyc_counts panel (NYC vs other PS imprint counts) onto ax."""
    years = counts.index.to_numpy()
    city_counts = counts["city"].to_numpy()
    other_counts = counts["other"].to_numpy()
    no_place_counts = counts["no_place"].to_numpy() if "no_place" in counts else None

    if args.ci:
        z = 1.96
        se_city = np.sqrt(city_counts)
        lower_city = (city_counts - z * se_city).clip(min=0)
        upper_city = city_counts + z * se_city
        ax.fill_between(
            years, lower_city, upper_city, color=style.COLOR_NYC, alpha=0.15
        )
        se_other = np.sqrt(other_counts)
        lower_other = (other_counts - z * se_other).clip(min=0)
        upper_other = other_counts + z * se_other
        ax.fill_between(
            years, lower_other, upper_other, color=style.COLOR_OTHER, alpha=0.15
        )
        if no_place_counts is not None:
            se_no_place = np.sqrt(no_place_counts)
            lower_no_place = (no_place_counts - z * se_no_place).clip(min=0)
            upper_no_place = no_place_counts + z * se_no_place
            ax.fill_between(
                years,
                lower_no_place,
                upper_no_place,
                color=style.COLOR_NOPLACE,
                alpha=0.15,
            )
    ax.plot(
        years,
        city_counts,
        label=args.city,
        color=style.COLOR_NYC,
        linestyle="-",
        marker=style.MARKERS[0],
        markevery=10,
        markersize=4,
    )
    ax.plot(
        years,
        other_counts,
        label="Other",
        color=style.COLOR_OTHER,
        linestyle="-",
        marker=style.MARKERS[1],
        markevery=10,
        markersize=4,
    )
    if no_place_counts is not None:
        ax.plot(
            years,
            no_place_counts,
            label="No place of publication",
            color=style.COLOR_NOPLACE,
            linestyle=":",
            marker=style.MARKERS[2],
            markevery=10,
            markersize=4,
        )
    ax.set_xlabel("Year")
    ax.set_ylabel("PS records with place of publication")
    ax.legend()


def compute_other_publisher_correlation(counts, publisher_counts) -> float:
    """Pearson correlation between ps_nyc_counts's 'Other' series and
    ps_unique_publishers's unique publisher counts, over the years the two
    series share."""
    other_series = counts["other"]
    common_years = other_series.index.intersection(publisher_counts.index)
    return other_series.loc[common_years].corr(publisher_counts.loc[common_years])


def plot_right(ax, publisher_counts, correlation: float | None = None) -> None:
    """Plot the ps_unique_publishers panel (unique PS publishers per year) onto ax."""
    years = publisher_counts.index.to_numpy()
    values = publisher_counts.to_numpy()

    ax.plot(years, values, **style.series_style(3), markevery=10, markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Unique PS publishers")
    ax.set_ylim(0, values.max() * 1.05)

    if correlation is not None:
        ax.annotate(
            f"r = {correlation:.2f} vs. Other (ps_nyc_counts)",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=9,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot a two-panel figure: NYC vs other PS imprint counts "
        "(left, from ps_nyc_counts) and unique PS publisher counts "
        "(right, from ps_unique_publishers)"
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
        help="City to highlight in the left panel (default: New York City)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for the figure (default: figures/outputs/nyc_and_publishers.png)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Include 95 pct confidence intervals for the left panel's series",
    )
    parser.add_argument(
        "--include-no-place",
        action="store_true",
        help='Plot "No place of publication" as its own series in the left panel',
    )
    parser.add_argument(
        "--annotate-correlation",
        action="store_true",
        help="Annotate the right panel with the correlation between unique PS "
        "publishers and ps_nyc_counts's 'Other' series",
    )
    args = parser.parse_args()

    df = ps_nyc_counts.load_data(args.input_csv)
    counts = ps_nyc_counts.compute_counts(
        df, args.city, include_no_place=args.include_no_place
    )
    publisher_counts = ps_unique_publishers.compute_unique_publishers(df)
    if publisher_counts.empty:
        raise ValueError(
            f"No publisher records between {ps_unique_publishers.YEAR_START} "
            f"and {ps_unique_publishers.YEAR_END}."
        )

    style.apply_style()
    base_width, base_height = plt.rcParams["figure.figsize"]
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(base_width * 2, base_height))

    correlation = None
    if args.annotate_correlation:
        correlation = compute_other_publisher_correlation(counts, publisher_counts)
        print(f"Correlation (Other vs. unique publishers): r = {correlation:.4f}")

    plot_left(ax_left, args, counts)
    plot_right(ax_right, publisher_counts, correlation=correlation)

    plt.tight_layout()
    style.save_figure(args.output)


if __name__ == "__main__":
    main()
