#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig4.png"
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


def compute_work_counts(df: pd.DataFrame) -> pd.Series:
    """
    Count works per year within the configured window.
    If LCCN is available, counts unique LCCNs and treats missing LCCNs as unique works.
    """
    filtered = df.loc[df["year_min"].between(YEAR_START, YEAR_END)].copy()
    filtered = filtered.dropna(subset=["year_min"])
    filtered["year_min"] = filtered["year_min"].astype(int)

    if "lccn" in filtered.columns:
        lccn_clean = filtered["lccn"].astype(str).str.strip().replace("", pd.NA)
        filtered = filtered.assign(lccn_clean=lccn_clean)
        counts = (
            filtered.dropna(subset=["lccn_clean"])
            .groupby("year_min")["lccn_clean"]
            .nunique()
        )
        missing = filtered[filtered["lccn_clean"].isna()].groupby("year_min").size()
        return counts.add(missing, fill_value=0)

    return filtered.groupby("year_min").size()


def compute_work_per_publisher(df: pd.DataFrame) -> pd.Series:
    """
    Compute works per year divided by unique publishers per year.
    """
    publishers = compute_unique_publishers(df)
    works = compute_work_counts(df)
    combined = pd.DataFrame({"publishers": publishers, "works": works}).fillna(0)
    combined = combined[combined["publishers"] > 0]
    combined = combined.sort_index()
    return combined["works"] / combined["publishers"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot unique PS works per unique publisher per year"
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
        help="Output file path for the figure (default: figures/outputs/fig4.png)",
    )
    args = parser.parse_args()

    works_per_publisher = compute_work_per_publisher(load_data(args.input_csv))
    years = works_per_publisher.index.to_numpy()
    values = works_per_publisher.to_numpy()

    if works_per_publisher.empty:
        raise ValueError(
            f"No publisher/work records between {YEAR_START} and {YEAR_END}."
        )

    style.apply_style()
    plt.figure()
    plt.plot(years, values)
    plt.xlabel("Year")
    plt.ylabel("Average PS works per PS publisher")
    ymax = values.max()
    upper = ymax * 1.05 if ymax > 0 else 1.0
    plt.ylim(0, upper)
    plt.tight_layout()
    style.save_figure(args.output)


if __name__ == "__main__":
    main()
