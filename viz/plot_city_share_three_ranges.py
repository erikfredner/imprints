#!/usr/bin/env python3
"""
Plot the percentage of PS, E, and F class imprints published in New York vs other locations.
Generates a comparative line chart over time.
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
) -> pd.Series:
    """
    Compute rolling-window percentage of records published in `city`.
    Returns a Series indexed by year with percentage for city (True flag).
    """
    df = df.copy()
    df['in_city'] = (df.get('city_group') == city).astype(int)
    mask = (df.get('year_min') >= start_year) & (df.get('year_min') <= end_year)
    grouped = (
        df[mask]
        .groupby(['year_min', 'in_city'])
        .size()
        .unstack(fill_value=0)
        .rolling(window, min_periods=1)
        .mean()
    )
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    return pct.get(1, pd.Series(0, index=grouped.index))

def main():
    parser = argparse.ArgumentParser(
        description="Compare New York share for PS, E, and F classes"
    )
    parser.add_argument(
        "--city",
        default="New York",
        help="City to highlight (default: New York)",
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
        "--window",
        type=int,
        default=5,
        help="Rolling window size (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="./viz/city_share_three_ranges.png",
        help="Output path for the comparative plot (default: ./viz/city_share_three_ranges.png)",
    )
    args = parser.parse_args()

    # Define classification ranges and input CSVs
    ranges = {
        'PS': './data/PS/data.csv',
        'E': './data/E/data.csv',
        'F': './data/F/data.csv',
    }
    # Compute share series
    shares = {}
    for cls, path in ranges.items():
        df = load_data(path)
        shares[cls] = compute_city_share(
            df,
            city=args.city,
            start_year=args.start_year,
            end_year=args.end_year,
            window=args.window,
        )
    # Prepare common year index
    years = range(args.start_year, args.end_year + 1)
    # Plot comparative line chart
    plt.figure(dpi=600)
    plt.style.use('tableau-colorblind10')
    for idx, (cls, series) in enumerate(shares.items()):
        pct = series.reindex(years, fill_value=0)
        plt.plot(years, pct.values, label=cls, color=f'C{idx}')
    plt.xlabel('Year')
    plt.ylabel(f'Percentage in {args.city}')
    plt.title(f'New York Share: PS vs E vs F (Rolling {args.window}-year)')
    plt.legend(title='Classification')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f'Saved figure to: {args.output}')

if __name__ == '__main__':
    main()