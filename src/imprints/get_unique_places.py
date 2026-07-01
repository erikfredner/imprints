"""
Extract sorted unique cleaned place names from a cleaned CSV.

By default this reports raw `places_clean` strings, which is what the
original curation of nyc_variants.txt was built from -- keep --canonical
off if reproducing or extending that workflow. NOTE: raw places_clean is
NOT a good estimate of *distinct physical places*: the same city fragments
across cataloging-convention variants (e.g. "boston", "boston ma", "boston
mass", "boston massachusetts" are 4 separate strings for one city), and
MARC "no place identified" markers ("s l", "n p") get counted as if they
were places. Pass --canonical to collapse US state-name variants via
imprints.place_canonicalization and drop those markers -- use this for any
count of "unique places of publication."

Usage:
    python -m imprints.get_unique_places --input_csv data/PS/data.csv --output_txt data/PS/places.txt
    python -m imprints.get_unique_places --input_csv data/PS/data.csv --output_txt data/PS/places_canonical.txt --canonical
"""

import argparse
import pandas as pd

from imprints.place_canonicalization import canonicalize_place


def get_unique_places(input_csv, output_txt, canonical=False):
    """Load cleaned CSV, collect unique non-empty places_clean values, and write to a text file.

    If canonical is True, collapse US state-name variants and drop known
    no-place markers via imprints.place_canonicalization.canonicalize_place
    before deduplicating.
    """
    df = pd.read_csv(input_csv)
    if "places_clean" not in df.columns:
        raise ValueError("Expected a 'places_clean' column in the input CSV.")

    places = df["places_clean"].dropna().astype(str).str.strip()
    places = places[places != ""]
    if canonical:
        places = places.map(canonicalize_place).dropna()
    unique_places = sorted({p for p in places if p})

    with open(output_txt, "w", encoding="utf-8") as f:
        for place in unique_places:
            f.write(f"{place}\n")

    return unique_places


def main():
    parser = argparse.ArgumentParser(
        description="Extract sorted unique place strings from a cleaned imprints CSV."
    )
    parser.add_argument(
        "--input_csv",
        required=True,
        help="Path to CSV output from imprints.data_cleaning",
    )
    parser.add_argument(
        "--output_txt",
        required=True,
        help="Path to write unique places (one per line).",
    )
    parser.add_argument(
        "--canonical",
        action="store_true",
        help=(
            "Collapse US state-name variants and drop 'no place identified' "
            "markers before deduplicating (see module docstring)."
        ),
    )
    args = parser.parse_args()

    unique_places = get_unique_places(args.input_csv, args.output_txt, args.canonical)
    print(f"Wrote {len(unique_places)} unique places to {args.output_txt}")


if __name__ == "__main__":
    main()
