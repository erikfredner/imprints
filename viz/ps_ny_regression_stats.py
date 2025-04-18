#!/usr/bin/env python3
"""
Compute PS imprint New York publication statistics and regression.

1. Identify the year (1800-2010) with the highest percentage of PS imprints
   published in New York.
2. Fit a linear regression (percentage vs. year) from 1800 up to that peak year.
3. Use the model to predict the percentage in a target year (default 2000).
4. Compare the predicted percentage to the actual percentage in that year.
"""
import argparse
import numpy as np
import pandas as pd


def load_data(csv_path: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Load and preprocess PS data CSV."""
    df = pd.read_csv(csv_path)
    # Ensure publication year is integer and within bounds
    df = df.copy()
    # Prefer the extracted publication year; fallback to year_min if needed
    year_col = 'year_int' if 'year_int' in df.columns else 'year_min'
    df['year'] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
    # Ensure city_group exists
    if 'city_group' not in df.columns:
        raise KeyError("city_group column not found in PS data CSV")
    return df


def compute_yearly_share(df: pd.DataFrame) -> pd.DataFrame:
    """Compute total and NYC share per year."""
    # Total PS imprints per year
    total = df.groupby('year').size().rename('total')
    # NYC imprints per year
    ny = df[df['city_group'] == 'New York'].groupby('year').size().rename('nyc')
    # Combine into DataFrame
    years = sorted(df['year'].unique())
    stats = pd.DataFrame(index=years)
    stats['total'] = total.reindex(years, fill_value=0)
    stats['nyc'] = ny.reindex(years, fill_value=0)
    # Percentage of NYC
    stats['pct_nyc'] = stats['nyc'] / stats['total'] * 100
    return stats


def fit_linear_regression(years: np.ndarray, pct: np.ndarray) -> tuple:
    """Fit linear model pct = intercept + slope * year. Return (intercept, slope, r2)."""
    # Fit slope and intercept
    slope, intercept = np.polyfit(years, pct, 1)
    # Predicted values
    pct_pred = intercept + slope * years
    # R^2 calculation
    ss_res = np.sum((pct - pct_pred) ** 2)
    ss_tot = np.sum((pct - np.mean(pct)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return intercept, slope, r2


def main():
    parser = argparse.ArgumentParser(
        description="PS New York imprint regression statistics"
    )
    parser.add_argument(
        "--input-csv",
        default="./data/PS/data.csv",
        help="Path to cleaned PS data CSV (from data_cleaning)"
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1800,
        help="Start year for analysis (inclusive)"
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2010,
        help="End year for analysis (inclusive)"
    )
    parser.add_argument(
        "--predict-year",
        type=int,
        default=2000,
        help="Year to predict NYC share using the regression model"
    )
    args = parser.parse_args()

    # Load and filter data
    df = load_data(args.input_csv, args.start_year, args.end_year)
    # Compute yearly stats
    stats = compute_yearly_share(df)

    # 1. Year with highest NYC percentage
    peak_year = int(stats['pct_nyc'].idxmax())
    peak_pct = stats.loc[peak_year, 'pct_nyc']
    print(f"Year with highest NYC share: {peak_year} ({peak_pct:.2f}%)")

    # Prepare data for regression: years up to peak_year
    reg_stats = stats.loc[stats.index <= peak_year]
    years_arr = reg_stats.index.values.astype(float)
    pct_arr = reg_stats['pct_nyc'].values.astype(float)

    # 2. Fit linear regression
    intercept, slope, r2 = fit_linear_regression(years_arr, pct_arr)
    print("\nLinear regression (pct_nyc = intercept + slope * year)")
    print(f"  Intercept: {intercept:.6f}")
    print(f"  Slope:     {slope:.6f}")
    print(f"  R^2:       {r2:.4f}")

    # 3. Predict for target year
    pred_year = args.predict_year
    # Standard linear regression prediction: pct = intercept + slope * year
    pred_pct = intercept + slope * pred_year
    print(f"\nPredicted NYC share in {pred_year}: {pred_pct:.2f}%")

    # 4. Actual NYC share in target year (if within range)
    if pred_year in stats.index:
        actual_pct = stats.loc[pred_year, 'pct_nyc']
        print(f"Actual NYC share in {pred_year}:    {actual_pct:.2f}%")
    else:
        print(f"Actual NYC share in {pred_year}:    N/A (year out of range)")

if __name__ == '__main__':
    main()