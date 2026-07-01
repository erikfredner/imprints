"""
Compute how many unique places of publication first appear after a cutoff year.

For each canonicalized place name, its "first appearance" is the minimum
``year_min`` across all rows resolving to that place. Reports the total
number of unique places, how many first appear after the cutoff, and that
count as a percentage of the total -- feeding a sentence like:

    "After normalizing place names, there are X unique places of publication
    in the PS subclassification, Y (Z%) of which first appear after 1960."

Raw ``places_clean`` values are NOT usable directly for this: they fragment
one city across cataloging-convention variants (e.g. "boston", "boston ma",
"boston mass", "boston massachusetts" are 4 distinct strings for one city)
and count MARC "no place identified" markers ("s l", "n p") as places. This
script canonicalizes via imprints.place_canonicalization before counting;
pass --raw to reproduce the uncorrected, inflated counts for comparison.

Usage:
    python -m imprints.first_appearance_stats --input-csv data/PS/data.csv --cutoff-year 1960
"""

import argparse

import pandas as pd

from imprints.place_canonicalization import canonicalize_place


def compute_first_appearance_stats(
    input_csv: str, cutoff_year: int, canonical: bool = True
) -> dict:
    """Return total unique places, count/pct first appearing after cutoff_year."""
    df = pd.read_csv(input_csv, usecols=["places_clean", "year_min"])

    places = df["places_clean"].dropna().astype(str).str.strip()
    df = df.assign(places_clean=places)
    df = df[df["places_clean"] != ""]
    df = df.dropna(subset=["year_min"])

    if canonical:
        df = df.assign(places_clean=df["places_clean"].map(canonicalize_place))
        df = df.dropna(subset=["places_clean"])

    first_year = df.groupby("places_clean")["year_min"].min()

    total = len(first_year)
    after_cutoff = int((first_year > cutoff_year).sum())
    pct = (after_cutoff / total * 100.0) if total else 0.0

    return {
        "total_unique_places": total,
        "after_cutoff": after_cutoff,
        "cutoff_year": cutoff_year,
        "pct_after_cutoff": pct,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute how many unique places of publication first appear "
            "after a cutoff year."
        )
    )
    parser.add_argument(
        "--input-csv",
        default="data/PS/data.csv",
        help="Path to the cleaned CSV (default: data/PS/data.csv).",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=1960,
        help="Year after which a place's first appearance counts (default: 1960).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Skip canonicalization and count raw places_clean strings "
            "(inflated by cataloging-format variants; see module docstring)."
        ),
    )
    args = parser.parse_args()

    stats = compute_first_appearance_stats(
        args.input_csv, args.cutoff_year, canonical=not args.raw
    )

    print(
        "After normalizing place names, there are "
        f"{stats['total_unique_places']} unique places of publication in the "
        f"PS subclassification, {stats['after_cutoff']} "
        f"({stats['pct_after_cutoff']:.1f}%) of which first appear after "
        f"{stats['cutoff_year']}."
    )


if __name__ == "__main__":
    main()
