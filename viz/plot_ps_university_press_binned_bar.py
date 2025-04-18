#!/usr/bin/env python3
"""
Plot stacked bar chart of PS-class imprints by "University Press" publishers
in 10-year bins (1900–2010), coloring bars by publisher name and grouping
small publishers into "Other University Presses".
"""
import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned PS imprint data from CSV."""
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(
        description="Stacked bar of PS imprints by University Press publishers (10-year bins)"
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
        "--bin-size",
        type=int,
        default=10,
        help="Width of year bins (default: 10)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top publishers to show (others grouped, default: 8)",
    )
    parser.add_argument(
        "--output",
        default="./viz/ps_university_press_binned_bar.png",
        help="Output path for the stacked bar chart (default: ./viz/ps_university_press_binned_bar.png)",
    )
    args = parser.parse_args()

    # Load and filter data
    df = load_data(args.input_csv)
    mask_year = (df.get("year_min") >= args.start_year) & (df.get("year_min") <= args.end_year)
    df = df.loc[mask_year]
    df = df[df.get("publisher_clean").notnull()]
    # Select University Press publishers
    up_mask = df["publisher_clean"].str.contains("university press", case=False, na=False)
    df_up = df.loc[up_mask].copy()
    # Title-case publisher names
    df_up['publisher_clean'] = df_up['publisher_clean'].str.title()

    # Identify top-N publishers overall (by total works)
    total_counts = df_up['publisher_clean'].value_counts()
    top_presses = total_counts.nlargest(args.top_n).index.tolist()
    # Group others
    df_up["press_group"] = np.where(
        df_up["publisher_clean"].isin(top_presses),
        df_up["publisher_clean"],
        "Other University Presses",
    )

    # Define bins (10-year intervals) and decade labels
    bins = list(range(args.start_year, args.end_year + 1, args.bin_size))
    bins.append(args.end_year + 1)
    # Labels: start of decade
    labels = [str(b) for b in bins[:-1]]
    # Assign bins
    df_up['bin'] = pd.cut(
        df_up['year_min'],
        bins=bins,
        labels=labels,
        right=False,
    )

    # Pivot counts
    grouped = (
        df_up
        .groupby(["bin", "press_group"]).size()
        .unstack(fill_value=0)
    )
    # Ensure bin order
    grouped = grouped.reindex(labels)
    # Order columns: top presses then Other
    cols = top_presses + ["Other University Presses"]
    cols = [c for c in cols if c in grouped.columns]
    data = grouped[cols]

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 6), dpi=600)
    plt.style.use('tableau-colorblind10')
    positions = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    for idx, press in enumerate(cols):
        ax.bar(
            positions,
            data[press].values,
            bottom=bottom,
            label=press,
            color=f'C{idx}',
        )
        bottom += data[press].values
    ax.set_xlabel('Year')
    ax.set_ylabel('Count of PS imprints')
    ax.set_title('PS imprints by University Presses (10-year bins)')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45)
    # Standard legend position
    ax.legend(title='Publisher', loc='upper right')
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=600)
    print(f'Saved stacked bar chart to: {args.output}')

if __name__ == "__main__":
    main()