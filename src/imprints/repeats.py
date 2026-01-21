"""
Compute the percentage of unique LCCNs that appear more than once in a dataset.

Usage:
    python -m imprints.repeats --input-csv data/PS/data.csv
"""

import argparse
import pandas as pd


def repeated_lccn_percentage(input_csv: str) -> float:
    """
    Return the percentage of unique LCCN values that occur more than once.
    """
    try:
        df = pd.read_csv(input_csv, usecols=["lccn"])
    except ValueError as exc:
        raise ValueError("Expected an 'lccn' column in the input CSV.") from exc

    lccns = (
        df["lccn"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    lccns = lccns[lccns != ""]

    if lccns.empty:
        return 0.0

    counts = lccns.value_counts()
    repeated = (counts > 1).sum()
    return (repeated / len(counts)) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the percentage of unique LCCNs that appear more than once "
            "in an imprints CSV."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/PS/data.csv",
        help="Path to the cleaned CSV (default: data/PS/data.csv).",
    )
    args = parser.parse_args()

    percentage = repeated_lccn_percentage(args.input_csv)
    print(f"{percentage:.2f}%")


if __name__ == "__main__":
    main()
