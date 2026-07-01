"""
Extract sorted unique cleaned place names from a cleaned CSV.

Usage:
    python -m imprints.get_unique_places --input_csv data/PS/data.csv --output_txt data/PS/places.txt
"""

import argparse
import pandas as pd


def get_unique_places(input_csv, output_txt):
    """Load cleaned CSV, collect unique non-empty places_clean values, and write to a text file."""
    df = pd.read_csv(input_csv)
    if "places_clean" not in df.columns:
        raise ValueError("Expected a 'places_clean' column in the input CSV.")

    places = df["places_clean"].dropna().astype(str).str.strip()
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
    args = parser.parse_args()

    unique_places = get_unique_places(args.input_csv, args.output_txt)
    print(f"Wrote {len(unique_places)} unique places to {args.output_txt}")


if __name__ == "__main__":
    main()
