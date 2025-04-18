#!/usr/bin/env python3
"""
Plot annual percentage of PS-class imprints published in New York
within specified classification ranges over time.
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
        description="Plot time series of NY share for PS subranges"
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
        default="./viz/ps_ranges_time_share.png",
        help="Output path for the line chart",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    # Ensure numeric class digits
    df = df[df['class_digits'].notnull()]
    df['class_digits'] = df['class_digits'].astype(int)
    # City flag
    df['in_city'] = (df.get('city_group') == args.city)
    # Filter year range
    df = df[(df['year_min'] >= args.start_year) & (df['year_min'] <= args.end_year)]

    # Define PS subranges
    ranges = {
        'Poetry (301-326)': (301, 326),
        'Drama (330-353)': (330, 353),
        'Prose (360-380)': (360, 380),
        'Essays (420-429)': (420, 429),
    }
    years = list(range(args.start_year, args.end_year + 1))
    ts = pd.DataFrame(index=years)
    for label, (low, high) in ranges.items():
        df_range = df[(df['class_digits'] >= low) & (df['class_digits'] <= high)]
        total = df_range.groupby('year_min').size()
        city = df_range[df_range['in_city']].groupby('year_min').size()
        share = (city / total * 100).fillna(0)
        ts[label] = share.reindex(years, fill_value=0)

    # Smooth with 5-year rolling mean
    ts_sm = ts.rolling(window=5, center=True, min_periods=1).mean()
    # Plot
    plt.figure(dpi=600)
    plt.style.use('tableau-colorblind10')
    for idx, col in enumerate(ts_sm.columns):
        plt.plot(years, ts_sm[col].values, label=col, color=f'C{idx}')
    plt.xlabel('Year')
    plt.ylabel(f'Percentage of PS imprints in {args.city}')
    plt.ylim(0, 100)
    plt.title('Annual % of PS imprints in New York by classification range')
    plt.legend(title='Range')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f"Saved share time series plot to: {args.output}")

if __name__ == '__main__':
    main()