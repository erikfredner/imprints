#!/usr/bin/env python3
"""
Plot total number of PS-class imprints published in New York
for specific classification ranges.
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
        description="Plot PS imprints counts in New York for selected ranges"
    )
    parser.add_argument(
        "--input-csv",
        default="./data/PS/data.csv",
        help="Path to cleaned PS data CSV",
    )
    parser.add_argument(
        "--city",
        default="New York",
        help="City to filter for (default: New York)",
    )
    parser.add_argument(
        "--output",
        default="./viz/ps_ranges_ny_counts.png",
        help="Output path for the counts bar chart",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    # Ensure numeric digits column
    df = df[df["class_digits"].notnull()]
    df["class_digits"] = df["class_digits"].astype(int)
    df["in_city"] = (df.get("city_group") == args.city)

    # Define ranges with labels
    ranges = {
        "PS301-326 Poetry": (301, 326),
        "PS330-353 Drama": (330, 353),
        "PS360-380 Prose": (360, 380),
        "PS420-429 Essays": (420, 429),
    }
    labels = []
    counts = []
    for label, (low, high) in ranges.items():
        mask = (df["class_digits"] >= low) & (df["class_digits"] <= high)
        total_in_city = df.loc[mask & df["in_city"]].shape[0]
        labels.append(label)
        counts.append(total_in_city)

    # Plot bar chart
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    x = range(len(labels))
    plt.bar(x, counts, color=[f"C{i}" for i in range(len(labels))])
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Count of PS imprints in New York")
    plt.title("PS imprints in New York by classification range")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f"Saved counts bar chart to: {args.output}")

if __name__ == "__main__":
    main()