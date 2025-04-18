#!/usr/bin/env python3
"""
Plot rolling correlation between unique publishers and PS imprint counts
within and outside New York City over time.

Also report average correlation coefficients to stdout.
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(csv_path: str, start_year: int, end_year: int):
    """Load cleaned PS data and filter by year range."""
    df = pd.read_csv(csv_path)
    # Ensure year_min exists and is integer
    if 'year_min' not in df.columns:
        raise KeyError("year_min column not found in PS data CSV")
    df = df.copy()
    df['year_min'] = pd.to_numeric(df['year_min'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['year_min'])
    df['year_min'] = df['year_min'].astype(int)
    # Filter to range
    return df[(df['year_min'] >= start_year) & (df['year_min'] <= end_year)]


def compute_annual_counts(df: pd.DataFrame, city: str, start_year: int, end_year: int):
    """Compute annual unique publisher counts and imprint counts by city flag."""
    # Unique publishers per year
    pub_counts = df.groupby('year_min')['publisher_clean'].nunique()
    # In-city flag
    df['in_city'] = (df.get('city_group') == city).astype(int)
    # Annual imprint counts in and out of city
    annual = df.groupby(['year_min', 'in_city']).size().unstack(fill_value=0)
    years = list(range(start_year, end_year + 1))
    pubs = pub_counts.reindex(years, fill_value=0)
    nyc = annual.get(1, pd.Series(dtype=int)).reindex(years, fill_value=0)
    other = annual.get(0, pd.Series(dtype=int)).reindex(years, fill_value=0)
    stats = pd.DataFrame({
        'pub_count': pubs,
        'nyc_count': nyc,
        'other_count': other,
    }, index=years)
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Plot rolling correlation of unique publishers with in/out City counts"
    )
    parser.add_argument(
        "--input-csv",
        default="./data/PS/data.csv",
        help="Path to cleaned PS data CSV",
    )
    parser.add_argument(
        "--city",
        default="New York",
        help="City to treat as in-city (default: New York)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1900,
        help="Start year (inclusive) for analysis",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2010,
        help="End year (inclusive) for analysis",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Window size (years) for rolling correlation (default: 5)",
    )
    parser.add_argument(
        "--output",
        default="./viz/ps_pub_loc_correlation.png",
        help="Output path for correlation plot",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv, args.start_year, args.end_year)
    stats = compute_annual_counts(
        df, args.city, args.start_year, args.end_year
    )
    # Compute rolling correlations
    corr_nyc = stats['pub_count'].rolling(
        window=args.window, center=True, min_periods=2
    ).corr(stats['nyc_count'])
    corr_other = stats['pub_count'].rolling(
        window=args.window, center=True, min_periods=2
    ).corr(stats['other_count'])

    # Compute standard errors for correlations using Fisher transformation approximation
    n = args.window
    if n > 3:
        denom = np.sqrt(n - 3)
        se_nyc = (1 - corr_nyc.pow(2)) / denom
        se_other = (1 - corr_other.pow(2)) / denom
    else:
        # insufficient window size for SE
        se_nyc = pd.Series(np.nan, index=stats.index)
        se_other = pd.Series(np.nan, index=stats.index)

    # Plot
    years = stats.index
    plt.figure(dpi=600)
    plt.style.use('tableau-colorblind10')
    # Plot correlations
    plt.plot(years, corr_nyc, label=f'Pub vs {args.city}', color='C0')
    plt.plot(years, corr_other, label='Pub vs Other', color='C1')
    # Shade ±1 SE around each correlation line
    plt.fill_between(years, corr_nyc - se_nyc, corr_nyc + se_nyc,
                     color='C0', alpha=0.2, linewidth=0)
    plt.fill_between(years, corr_other - se_other, corr_other + se_other,
                     color='C1', alpha=0.2, linewidth=0)
    plt.xlabel('Year')
    plt.ylabel('Correlation coefficient')
    plt.title(f'Rolling {args.window}-year correlation')
    plt.legend()
    plt.ylim(-1, 1)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=600)
    print(f"Saved correlation plot to: {args.output}")

    # Report average correlations
    avg_nyc = corr_nyc.mean()
    avg_other = corr_other.mean()
    # Average standard errors
    avg_se_nyc = se_nyc.mean()
    avg_se_other = se_other.mean()
    print(f"Average rolling ({args.window}-yr) corr Pub vs {args.city}: {avg_nyc:.4f}")
    print(f"Average rolling ({args.window}-yr) corr Pub vs Other:  {avg_other:.4f}")
    print(f"Average rolling SE Pub vs {args.city}: {avg_se_nyc:.4f}")
    print(f"Average rolling SE Pub vs Other:  {avg_se_other:.4f}")

if __name__ == '__main__':
    main()