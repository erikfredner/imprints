#!/usr/bin/env python3
"""
Generate an annual line chart of absolute counts of PS-class imprints
published in New York vs other locations (1900–2010).
"""
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Plot absolute PS imprint counts in New York vs other locations"
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
        "--output-line",
        default="./viz/ps_counts_line.png",
        help="Output path for line chart (default: ./viz/ps_counts_line.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    # Filter to fixed year range 1900–2010
    mask = (df.get("year_min") >= 1900) & (df.get("year_min") <= 2010)
    df = df[mask].copy()
    # Flag in-city vs other
    df["in_city"] = (df.get("city_group") == args.city).astype(int)

    # Annual counts
    annual = (
        df.groupby(["year_min", "in_city"]) .size() .unstack(fill_value=0)
    )
    years = annual.index.values
    other_counts = annual.get(0, pd.Series(0, index=years))
    city_counts = annual.get(1, pd.Series(0, index=years))

    # Plot line chart: New York first for consistent color (C0), then Other (C1)
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    # Plot New York counts first to assign C0
    plt.plot(years, city_counts, label=args.city, color="C0")
    # Plot other locations counts second to assign C1
    plt.plot(years, other_counts, label="Other locations", color="C1")
    plt.xlabel("Year")
    plt.ylabel("Count of PS imprints")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_line), exist_ok=True)
    plt.savefig(args.output_line, dpi=600)
    print(f"Saved line chart to: {args.output_line}")

if __name__ == "__main__":
    main()