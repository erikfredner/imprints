#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig3.png"
YEAR_START = 1900
YEAR_END = 2010


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load cleaned PS imprint data from CSV."""
    return pd.read_csv(csv_path)


def compute_unique_publishers(df: pd.DataFrame) -> pd.Series:
    """
    Count unique cleaned publishers per year within the configured window.
    Ensures the index is integer for plotting.
    """
    filtered = df.loc[df["year_min"].between(YEAR_START, YEAR_END)].copy()
    filtered = filtered.dropna(subset=["publisher_clean"])
    filtered["year_min"] = filtered["year_min"].astype(int)
    return filtered.groupby("year_min")["publisher_clean"].nunique()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot unique publisher counts per year for PS imprints"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: figures/outputs/fig3.png)",
    )
    args = parser.parse_args()

    counts = compute_unique_publishers(load_data(args.input_csv))
    years = counts.index.to_numpy()
    values = counts.to_numpy()

    if counts.empty:
        raise ValueError(f"No publisher records between {YEAR_START} and {YEAR_END}.")

    style.apply_style()
    plt.figure()
    plt.plot(years, values)
    plt.xlabel("Year")
    plt.ylabel("Unique PS publishers")
    ymax = values.max()
    plt.ylim(0, ymax * 1.05)
    plt.tight_layout()
    style.save_figure(args.output)


if __name__ == "__main__":
    main()
