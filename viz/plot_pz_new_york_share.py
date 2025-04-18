#!/usr/bin/env python3
"""
Generate a stacked area plot showing the percentage of PZ-class imprints
published in New York vs. other locations over time.
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
) -> pd.DataFrame:
    """
    Compute rolling-window percentage of records published in `city`.
    Returns a DataFrame indexed by year_min with percentage of city.
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
    return pct


def plot_area(
    pct_df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
    window: int,
    output_path: str = None,
) -> None:
    """Plot a stacked area chart of city vs other locations and save or show."""
    plt.style.use('tableau-colorblind10')
    fig, ax = plt.subplots(figsize=(8, 4), dpi=600)
    pct_city = pct_df.get(1)
    years = pct_city.index
    ax.plot(years, pct_city, color='C0')
    ax.axhline(50, color='gray', linestyle='--', linewidth=1)
    # first crossing above 50%
    above = pct_city > 50
    if above.any():
        year_up = int(pct_city[above].index[0])
        val_up = pct_city.loc[year_up]
        ax.annotate(str(year_up), xy=(year_up, val_up), xytext=(year_up - window, min(val_up + 5, 100)),
                    ha='right', va='bottom', arrowprops=dict(arrowstyle='->', color='gray'))
        # first drop below
        after_up = pct_city.loc[pct_city.index > year_up]
        below_up = after_up < 50
        if below_up.any():
            year_down = int(after_up[below_up].index[0])
            val_down = pct_city.loc[year_down]
            ax.annotate(str(year_down), xy=(year_down, val_down), xytext=(year_down + window, max(val_down - 5, 0)),
                        ha='left', va='top', arrowprops=dict(arrowstyle='->', color='gray'))
            # lowest after drop
            after_down = pct_city.loc[pct_city.index > year_down]
            if not after_down.empty:
                year_low = int(after_down.idxmin())
                val_low = after_down.min()
                ax.annotate(str(year_low), xy=(year_low, val_low), xytext=(year_low, min(val_low + 5, 100)),
                            ha='center', va='bottom', arrowprops=dict(arrowstyle='->', color='gray'))
                # peak after low
                post_low = after_down.loc[after_down.index > year_low]
                if not post_low.empty:
                    year_peak = int(post_low.idxmax())
                    val_peak = post_low.max()
                    ax.annotate(str(year_peak), xy=(year_peak, val_peak), xytext=(year_peak, min(val_peak + 5, 100)),
                                ha='center', va='bottom', arrowprops=dict(arrowstyle='->', color='gray'))
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Percentage in {city}')
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=600)
        print(f'Saved figure to: {output_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot PZ-class imprints published in New York vs other locations"
    )
    parser.add_argument(
        '--input-csv',
        default='./data/PZ/data.csv',
        help='Path to cleaned PZ data CSV (default: ./data/PZ/data.csv)',
    )
    parser.add_argument(
        '--city',
        default='New York',
        help='City to highlight (default: New York)',
    )
    parser.add_argument('--start-year', type=int, default=1900, help='Start year')
    parser.add_argument('--end-year', type=int, default=2010, help='End year')
    parser.add_argument('--window', type=int, default=5, help='Rolling window size')
    parser.add_argument(
        '--output',
        default='./viz/pz_new_york_share.png',
        help='Output file path for PZ New York share plot (default: ./viz/pz_new_york_share.png)',
    )
    args = parser.parse_args()
    df = load_data(args.input_csv)
    pct = compute_city_share(df, args.city, args.start_year, args.end_year, args.window)
    plot_area(pct, args.city, args.start_year, args.end_year, args.window, args.output)

if __name__ == '__main__':
    main()