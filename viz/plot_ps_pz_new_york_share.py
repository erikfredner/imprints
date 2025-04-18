#!/usr/bin/env python3
"""
Plot the percentage of PS and PZ class imprints published in New York vs other locations,
combining deduplicated records from both ranges.
"""
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)

def compute_city_share(df: pd.DataFrame, city: str, start_year: int, end_year: int, window: int) -> pd.Series:
    """
    Compute rolling-window percentage of records published in `city`.
    Returns a Series indexed by year_min with percentage of city.
    """
    df = df.copy()
    # Flag city
    df['in_city'] = (df.get('city_group') == city).astype(int)
    # Filter years
    df = df[df['year_min'].between(start_year, end_year)]
    # Group and smooth
    grouped = (
        df.groupby(['year_min', 'in_city'])
          .size()
          .unstack(fill_value=0)
          .rolling(window, min_periods=1)
          .mean()
    )
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    return pct.get(1, pd.Series(0, index=grouped.index))

def main():
    parser = argparse.ArgumentParser(
        description="Combine PS and PZ and plot NYC share over time"
    )
    parser.add_argument('--ps-csv', default='./data/PS/data.csv', help='PS cleaned data CSV')
    parser.add_argument('--pz-csv', default='./data/PZ/data.csv', help='PZ cleaned data CSV')
    parser.add_argument('--city', default='New York', help='City to highlight')
    parser.add_argument('--start-year', type=int, default=1900, help='Start year (default: 1900)')
    parser.add_argument('--end-year', type=int, default=2010, help='End year (default: 2010)')
    parser.add_argument('--window', type=int, default=5, help='Rolling window (default: 5)')
    parser.add_argument('--output', default='./viz/ps_pz_new_york_share.png', help='Output plot path')
    args = parser.parse_args()

    # Load and combine, deduplicate
    df_ps = load_data(args.ps_csv)
    df_pz = load_data(args.pz_csv)
    df_all = pd.concat([df_ps, df_pz], ignore_index=True)
    if 'lccn' in df_all.columns:
        df_all = df_all.drop_duplicates(subset=['lccn'])

    # Compute city share
    pct_series = compute_city_share(
        df_all, args.city, args.start_year, args.end_year, args.window
    )
    years = pct_series.index.values

    # Plot
    plt.figure(dpi=600)
    plt.style.use('tableau-colorblind10')
    plt.plot(years, pct_series.values, color='C0')
    plt.axhline(50, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Year')
    plt.ylabel(f'Percentage in {args.city}')
    plt.title(f'Combined PS & PZ imprints in {args.city} (Rolling {args.window}-year)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f'Saved combined share plot to: {args.output}')

if __name__ == '__main__':
    main()