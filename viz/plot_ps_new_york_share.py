#!/usr/bin/env python3
"""
Generate a stacked area plot showing the percentage of PS-class imprints
published in New York vs. other locations over time.
"""
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def compute_city_share(
    df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
    window: int,
) -> pd.DataFrame:
    """
    Compute annual percentage of records published in `city` (no smoothing).
    Returns a DataFrame indexed by year with two columns:
    other locations and city percentages.
    """
    # Flag rows published in the target city
    df["in_city"] = (df["city_group"] == city).astype(int)
    # Filter to desired year range
    mask = (df["year_min"] >= start_year) & (df["year_min"] <= end_year)
    # Group by year and city flag, count occurrences (no smoothing)
    counts = df[mask].groupby(["year_min", "in_city"])\
                     .size()\
                     .unstack(fill_value=0)
    # Convert counts to percentages
    pct = counts.div(counts.sum(axis=1), axis=0) * 100
    return pct


def plot_area(
    pct_df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
    window: int,
    output_path: str = None,
) -> None:
    """Plot a stacked area chart of city vs other locations and save or show."""
    # Use colorblind-friendly palette
    plt.style.use("tableau-colorblind10")
    fig, ax = plt.subplots(dpi=600)
    # Plot percentage of imprints in target city as line + filled area
    # Column '1' is the True (in_city) percentage
    pct_city = pct_df.get(1)
    years = pct_city.index
    # Plot percentage of imprints in target city as a line
    ax.plot(years, pct_city, color="C0")
    # 50% reference line
    ax.axhline(50, color="gray", linestyle="--", linewidth=1)
    # Annotations: first and last crossings above 50%, then lowest and highest after
    above = pct_city > 50
    if above.any():
        years_above = pct_city[above].index
        # First crossing above 50%
        year_first = int(years_above[0])
        val_first = pct_city.loc[year_first]
        ax.annotate(
            str(year_first),
            xy=(year_first, val_first),
            xytext=(year_first - window, min(val_first + 5, 100)),
            ha="right",
            va="bottom",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )
        # Last time above 50%
        year_last = int(years_above[-1])
        val_last = pct_city.loc[year_last]
        if len(years_above) > 1:
            ax.annotate(
                str(year_last),
                xy=(year_last, val_last),
                xytext=(year_last + window, min(val_last + 5, 100)),
                ha="left",
                va="bottom",
                arrowprops=dict(arrowstyle="->", color="gray"),
            )
        # Third: lowest percentage after last crossing above 50%
        after_last = pct_city.loc[pct_city.index > year_last]
        if not after_last.empty:
            year_low = int(after_last.idxmin())
            val_low = after_last.min()
            ax.annotate(
                str(year_low),
                xy=(year_low, val_low),
                xytext=(year_low - window, min(val_low + 5, 100)),
                ha="right",
                va="bottom",
                arrowprops=dict(arrowstyle="->", color="gray"),
            )
            # Fourth: highest peak after the lowest point
            after_low = pct_city.loc[pct_city.index > year_low]
            if not after_low.empty:
                year_peak = int(after_low.idxmax())
                val_peak = after_low.max()
                ax.annotate(
                    str(year_peak),
                    xy=(year_peak, val_peak),
                    xytext=(year_peak + window, min(val_peak + 5, 100)),
                    ha="left",
                    va="bottom",
                    arrowprops=dict(arrowstyle="->", color="gray"),
                )
    ax.set_xlabel("Year")
    ax.set_ylabel("Percentage in " + city)
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"Saved figure to: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot PS imprints published in New York vs. other locations"
    )
    parser.add_argument(
        "--input-csv",
        default="./data/PS/data.csv",
        help="Path to cleaned PS data CSV (default: ./data/PS/data.csv)",
    )
    parser.add_argument(
        "--city",
        default="New York",
        help="City to highlight (default: New York)",
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
        help="Rolling window size in years (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="./viz/ps_new_york_share.png",
        help="Output file path for the figure (default: ./viz/ps_new_york_share.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    pct = compute_city_share(
        df,
        city=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
    )
    plot_area(
        pct,
        city=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
