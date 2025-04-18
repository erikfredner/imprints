#!/usr/bin/env python3
"""
Plot annual counts of PS-class imprints published by publishers whose cleaned
names contain 'university press', over 1900–2010.
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
        description="Plot PS imprints counts for 'university press' publishers"
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
        default="./viz/ps_university_press_counts.png",
        help="Output file path for the figure (default: ./viz/ps_university_press_counts.png)",
    )
    args = parser.parse_args()

    # Load and filter data
    df = load_data(args.input_csv)
    mask_year = (df.get("year_min") >= args.start_year) & (df.get("year_min") <= args.end_year)
    df = df.loc[mask_year]
    # Use cleaned publisher names
    df = df[df.get("publisher_clean").notnull()]
    # Filter for 'university press'
    mask_up = df["publisher_clean"].str.contains("university press", case=False, na=False)
    df_up = df.loc[mask_up]

    # Count works per year
    counts = df_up.groupby("year_min").size()
    # Count unique university presses per year
    unique_counts = df_up.groupby("year_min")["publisher_clean"].nunique()
    # Ensure all years present
    yr_idx = list(range(args.start_year, args.end_year + 1))
    counts = counts.reindex(yr_idx, fill_value=0)
    unique_counts = unique_counts.reindex(yr_idx, fill_value=0)
    years = counts.index.values
    works = counts.values
    uniques = unique_counts.values

    # Plot line: works and unique presses
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    plt.plot(years, works, color="C2", label="Works by Univ. Presses")
    plt.plot(years, uniques, color="C3", linestyle="--", label="Unique Univ. Presses")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.title("PS imprints by 'University Press': Works vs Unique Presses (1900–2010)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f"Saved university press counts plot to: {args.output}")


if __name__ == "__main__":
    main()