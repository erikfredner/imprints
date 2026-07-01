#!/usr/bin/env python3
"""Plot unique places of publication per 100 PS records, by year (1900-2010).

A type-token diversity metric: how many *distinct* cleaned place names
(``places_clean``) appear per 100 PS records in a given year. Complements the
NYC-share figures by asking whether publishing became more geographically
dispersed over time, independent of NYC's share specifically.

Record counts are deduplicated by ``lccn`` (see ``compute_record_counts``),
mirroring ``fig4.py::compute_work_counts`` -- the pipeline explodes one source
record into multiple rows when it lists multiple places, so a raw row count
would overstate "records".
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1] / "outputs/unique_places_per_100.png"
)
YEAR_START = 1900
YEAR_END = 2010


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load cleaned PS imprint data from CSV."""
    return pd.read_csv(csv_path, usecols=["year_min", "places_clean", "lccn"])


def compute_record_counts(df: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Unique PS records per year, deduplicated by ``lccn``.

    Rows with a missing/blank ``lccn`` each count as their own record (a
    record can't be deduped against others it can't be matched to). Mirrors
    ``fig4.py::compute_work_counts``, which corrects for the same
    explode-on-places row inflation.
    """
    filtered = df.loc[df["year_min"].between(start, end)].copy()
    filtered["year_min"] = filtered["year_min"].astype(int)
    lccn_clean = filtered["lccn"].astype(str).str.strip().replace("", pd.NA)
    filtered = filtered.assign(lccn_clean=lccn_clean)
    counts = (
        filtered.dropna(subset=["lccn_clean"])
        .groupby("year_min")["lccn_clean"]
        .nunique()
    )
    missing = filtered[filtered["lccn_clean"].isna()].groupby("year_min").size()
    return counts.add(missing, fill_value=0)


def compute_unique_places(df: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Count of distinct ``places_clean`` values per year."""
    filtered = df.loc[df["year_min"].between(start, end)].copy()
    filtered = filtered.dropna(subset=["places_clean"])
    filtered["year_min"] = filtered["year_min"].astype(int)
    return filtered.groupby("year_min")["places_clean"].nunique()


def compute_rate(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Annual unique places, record counts, and unique places per 100 records."""
    records = compute_record_counts(df, start, end)
    unique_places = compute_unique_places(df, start, end)
    years = pd.Index(range(start, end + 1), name="year_min")
    combined = pd.DataFrame(
        {"records": records, "unique_places": unique_places}
    ).reindex(years, fill_value=0)
    combined["rate"] = (
        combined["unique_places"] / combined["records"].replace(0, pd.NA) * 100
    )
    return combined


def plot(rate_df: pd.DataFrame, output: Path) -> None:
    """Line chart of the annual rate."""
    style.apply_style()
    plt.figure()
    plt.plot(rate_df.index.to_numpy(), rate_df["rate"].to_numpy())
    plt.xlabel("Year")
    plt.ylabel("Unique places of publication per 100 PS records")
    plt.tight_layout()
    style.save_figure(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot unique places of publication per 100 PS records, by year"
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
        help="Output file path for the figure (default: figures/outputs/unique_places_per_100.png)",
    )
    parser.add_argument("--start-year", type=int, default=YEAR_START)
    parser.add_argument("--end-year", type=int, default=YEAR_END)
    args = parser.parse_args()

    df = load_data(args.input_csv)
    rate_df = compute_rate(df, args.start_year, args.end_year)

    plot(rate_df, args.output)

    series_csv = args.output.with_name(f"{args.output.stem}_series.csv")
    rate_df.to_csv(series_csv, index_label="year")
    print(f"Saved annual series to: {series_csv}")

    def window_average(start: int, end: int) -> float:
        window = rate_df.loc[rate_df.index.to_series().between(start, end), "rate"]
        return window.mean()

    print(
        "Average unique places per 100 PS records, 1900-1950: "
        f"{window_average(1900, 1950):.2f}"
    )
    print(
        "Average unique places per 100 PS records, 1950-2000: "
        f"{window_average(1950, 2000):.2f}"
    )


if __name__ == "__main__":
    main()
