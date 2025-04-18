#!/usr/bin/env python3
"""
Plot annual counts and percentage share of publications in New York (1800-2010)
using data from MDS_pub_locations.csv.

Generates two line charts:
- Absolute number of records published in New York each year.
- Percentage of records published in New York each year.
"""
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_path: str) -> pd.DataFrame:
    """Load publication location data."""
    df = pd.read_csv(csv_path)
    # Normalize data types
    df = df.copy()
    # Year to integer
    df['year_of_publication'] = pd.to_numeric(
        df['year_of_publication'], errors='coerce'
    ).astype('Int64')
    df = df.dropna(subset=['year_of_publication'])
    df['year_of_publication'] = df['year_of_publication'].astype(int)
    # Convert is_new_york to boolean
    df['is_new_york'] = (
        df['is_new_york'].astype(str).str.lower() == 'true'
    )
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Plot time series of New York publication counts and share."
    )
    parser.add_argument(
        "--input-csv",
        default="./data/MDS_pub_locations.csv",
        help="Path to MDS publication locations CSV",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1800,
        help="Start year (inclusive, default: 1800)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2010,
        help="End year (inclusive, default: 2010)",
    )
    parser.add_argument(
        "--output-counts",
        default="./viz/mds_ny_time_counts.png",
        help="Output path for counts line chart",
    )
    parser.add_argument(
        "--output-share",
        default="./viz/mds_ny_time_share.png",
        help="Output path for share line chart",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    # Filter year range
    df = df[(df['year_of_publication'] >= args.start_year) &
            (df['year_of_publication'] <= args.end_year)]

    # Prepare year index
    years = list(range(args.start_year, args.end_year + 1))
    # Total publications per year
    total = df.groupby('year_of_publication').size().reindex(years, fill_value=0)
    # New York publications per year
    ny = df[df['is_new_york']].groupby('year_of_publication').size().reindex(years, fill_value=0)
    # Percentage share
    share = (ny / total * 100).fillna(0)

    # Plot counts
    plt.figure(dpi=600)
    plt.style.use('tableau-colorblind10')
    plt.plot(years, ny.values, label='New York', color='C0')
    plt.xlabel('Year')
    plt.ylabel('Number of publications')
    plt.title(f'Annual number of publications in New York ({args.start_year}-{args.end_year})')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_counts), exist_ok=True)
    plt.savefig(args.output_counts, dpi=600)
    print(f"Saved counts plot to: {args.output_counts}")

    # Plot share
    plt.figure(dpi=600)
    plt.style.use('tableau-colorblind10')
    plt.plot(years, share.values, label='New York share', color='C1')
    plt.xlabel('Year')
    plt.ylabel('Percentage of publications in New York')
    plt.ylim(0, 100)
    plt.title(f'Annual % of publications in New York ({args.start_year}-{args.end_year})')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_share), exist_ok=True)
    plt.savefig(args.output_share, dpi=600)
    print(f"Saved share plot to: {args.output_share}")

if __name__ == '__main__':
    main()