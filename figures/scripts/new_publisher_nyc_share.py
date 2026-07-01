#!/usr/bin/env python3
"""
Generate a line chart of New York City's share of publisher-place-of-publication
pairings appearing in the PS dataset for the first time each year.

Tests the assumption behind fig3's rising unique-publisher count: that growth is
happening mostly outside NYC. Unlike fig1 (NYC's share of all PS records each year),
this tracks where *newly appearing* publisher-place pairings are located, since a
publisher name alone isn't a stable unit (the same name could relocate).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import fig1
import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "outputs/new_publisher_nyc_share.png"
)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def compute_first_occurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce to one row per unique (publisher_clean, places_clean) pairing, with the
    year the pairing first appears anywhere in the dataset and its city_group.

    Runs on the full, unrestricted dataframe (no year-window filter) so a pairing's
    true first appearance is captured even outside the eventual display window;
    only the caller's display/report step restricts to a year range. Rows lacking a
    publisher or place, or with no place of publication, are dropped first, matching
    fig1's NO_PLACE handling.
    """
    df = df.dropna(subset=["publisher_clean", "places_clean"])
    df = df[df["city_group"] != fig1.NO_PLACE]
    pairs = df.groupby(["publisher_clean", "places_clean"], as_index=False).agg(
        year_min=("year_min", "min"), city_group=("city_group", "first")
    )
    return pairs


def compute_peak_share_year(
    df: pd.DataFrame, city: str, start_year: int, end_year: int
) -> int:
    """Year within [start_year, end_year] with the highest (unsmoothed) share of
    all PS records in `city`, per fig1.compute_city_share."""
    pct = fig1.compute_city_share(
        df, city=city, start_year=start_year, end_year=end_year, smooth=False
    )
    return int(pct[1].idxmax())


def compute_period_probability(
    pairs: pd.DataFrame, city: str, start_year: int, split_year: int, end_year: int
) -> tuple[float, float, int, int]:
    """
    Pooled probability that a new-to-PS pairing is in `city`, before vs. from
    `split_year` onward (within [start_year, end_year]). Pooling raw pairs (rather
    than averaging the per-year, possibly-smoothed share) weights each pairing
    equally regardless of how many others debuted the same year.
    """
    windowed = pairs[pairs["year_min"].between(start_year, end_year)]
    before = windowed[windowed["year_min"] < split_year]
    after = windowed[windowed["year_min"] >= split_year]
    prob_before = (before["city_group"] == city).mean() if len(before) else float("nan")
    prob_after = (after["city_group"] == city).mean() if len(after) else float("nan")
    return prob_before, prob_after, len(before), len(after)


def print_period_probability(
    pairs: pd.DataFrame, city: str, start_year: int, split_year: int, end_year: int
) -> None:
    """Print the before/after probability (and equivalent odds) that a new-to-PS
    pairing is in `city`, split at `split_year`."""
    prob_before, prob_after, n_before, n_after = compute_period_probability(
        pairs, city, start_year, split_year, end_year
    )
    odds_before = prob_before / (1 - prob_before)
    odds_after = prob_after / (1 - prob_after)

    print(
        f"\nBefore {split_year}, the probability that a publisher new to PS was "
        f"located in {city} was {prob_before:.1%} (N={n_before:,}). "
        f"From {split_year} onward, it falls to {prob_after:.1%} (N={n_after:,})."
    )
    print(
        f"Same logic as odds: before {split_year}, the odds a new pairing was in "
        f"{city} were {odds_before:.2f}:1 (1 {city} pairing per "
        f"{1 / odds_before:.1f} elsewhere). From {split_year} onward, the odds "
        f"fall to {odds_after:.2f}:1 (1 per {1 / odds_after:.1f} elsewhere)."
    )
    print(
        f"Odds ratio (before vs. after {split_year}): {odds_before / odds_after:.2f}x"
    )
    print(
        f"Odds ratio (after vs. before {split_year}): {odds_after / odds_before:.2f}x "
        f"(odds of a new pairing being in {city} fell to "
        f"{odds_after / odds_before:.1%} of their pre-{split_year} level)"
    )


def plot_share(
    pct_df: pd.DataFrame,
    city: str,
    output_path: str = None,
) -> None:
    """Plot a line chart of NYC's share of new-to-PS pairings and save or show."""
    style.apply_style()
    fig, ax = plt.subplots()
    pct_city = pct_df[1]
    years = pct_city.index.values

    ax.plot(years, pct_city, color=style.COLOR_NYC)
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)

    ax.set_xlabel("Year")
    ax.set_ylabel(f"Share of new-to-PS\npublisher-place pairs in {city}")
    style.percent_yaxis(ax)
    plt.tight_layout()
    if output_path:
        style.save_figure(output_path)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot NYC's share of publisher-place pairings new to PS each year"
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
        "--split-year",
        type=int,
        default=None,
        help="Year at which to split the before/after probability and odds "
        "summary printed to stdout (default: the year of peak NYC share of all "
        "PS records, per fig1.compute_city_share)",
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: "
        "figures/outputs/new_publisher_nyc_share.png)",
    )
    args = parser.parse_args()

    city = "New York City" if args.city == "New York" else args.city

    df = load_data(args.input_csv)
    pairs = compute_first_occurrence(df)

    print(f"Unique publisher-place pairings: {len(pairs):,}")
    print(pairs["city_group"].value_counts().to_string())

    split_year = args.split_year
    if split_year is None:
        split_year = compute_peak_share_year(
            df, city=city, start_year=args.start_year, end_year=args.end_year
        )
        print(f"Peak NYC share year (used as split year): {split_year}")

    print_period_probability(pairs, city, args.start_year, split_year, args.end_year)

    pct = fig1.compute_city_share(
        pairs,
        city=city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        smooth=args.smooth,
    )

    pct_city = pct[1]
    pct_rounded = pct_city.round()

    first_below_50 = pct_rounded[pct_rounded < 50]
    year_first_below_50 = (
        int(first_below_50.index.min()) if not first_below_50.empty else None
    )
    year_lowest = int(pct_city.idxmin()) if not pct_city.empty else None
    pct_lowest = float(pct_city.loc[year_lowest]) if year_lowest is not None else None
    year_highest = int(pct_city.idxmax()) if not pct_city.empty else None
    pct_highest = (
        float(pct_city.loc[year_highest]) if year_highest is not None else None
    )

    print(
        "First year NYC share of new pairings falls below 50% (rounded):",
        year_first_below_50,
    )
    print("Year with lowest NYC share of new pairings:", year_lowest)
    print(
        "NYC share in that year (%):",
        None if pct_lowest is None else round(pct_lowest, 2),
    )
    print("Year with highest NYC share of new pairings:", year_highest)
    print(
        "NYC share in that year (%):",
        None if pct_highest is None else round(pct_highest, 2),
    )

    plot_share(pct, city=city, output_path=args.output)


if __name__ == "__main__":
    main()
