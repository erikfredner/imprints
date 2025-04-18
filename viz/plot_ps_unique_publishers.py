#!/usr/bin/env python3
"""
Plot the raw count of unique cleaned publishers per year for PS-class imprints.
"""
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned PS imprint data from CSV."""
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot unique publisher counts per year for PS imprints"
    )
    parser.add_argument(
        "--input-csv",
        default="./data/PS/data.csv",
        help="Path to cleaned PS data CSV (default: ./data/PS/data.csv)",
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
        "--output",
        default="./viz/ps_unique_publishers.png",
        help="Output file path for the figure (default: ./viz/ps_unique_publishers.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    # Filter to year range
    mask = (df.get("year_min") >= args.start_year) & (df.get("year_min") <= args.end_year)
    df = df.loc[mask]
    # Drop records with missing publisher_clean
    df = df[df.get("publisher_clean").notnull()]

    # Count unique publishers per year (no smoothing)
    counts = df.groupby("year_min")["publisher_clean"].nunique()
    years = counts.index.astype(int)
    values = counts.values

    # Plot unique publisher counts with consistent color C2
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    plt.plot(years, values, color="C2", linewidth=2)
    plt.xlabel("Year")
    plt.ylabel("Unique publishers")
    # Add headroom at top
    ymin = values.min()
    ymax = values.max()
    plt.ylim(ymin, ymax * 1.05)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f"Saved unique publisher plot to: {args.output}")


if __name__ == "__main__":
    main()