"""
Count records in the raw record pickles that fall in LC class P and subclass PS.

The ``data/PS`` pickles retain *all* records from the raw MARC XML (see
``imprints.cross_range``), so no re-collection is needed. This reuses the
same prefix-matching logic as the rest of the pipeline
(``imprints.data_cleaning.matches_range``/``filter_classifications``) so the
counts are consistent with how PS records are identified everywhere else. A
record counts once toward a class if *any* of its (possibly multiple)
classifications match, mirroring ``cleaning_pipeline``'s filter step -- this
is a raw record count, not exploded by place like the cleaned CSV.

Usage:
    python -m imprints.class_totals --input_dir data/PS
"""

import argparse

from imprints.data_cleaning import (
    filter_classifications,
    load_pickles_to_dataframe,
    parse_range_spec,
)


def count_matching(df, range_spec):
    class_range = parse_range_spec(range_spec)
    matches = df["classifications"].map(
        lambda clist: len(filter_classifications(clist, class_range)) > 0
    )
    return int(matches.sum())


def main():
    parser = argparse.ArgumentParser(
        description="Count records in LC class P and subclass PS."
    )
    parser.add_argument(
        "--input_dir",
        default="data/PS",
        help="Directory of record pickles holding ALL classes (default: data/PS)",
    )
    args = parser.parse_args()

    print(f"Loading record pickles from: {args.input_dir}")
    df = load_pickles_to_dataframe(args.input_dir)
    print(f"Loaded {len(df):,} total records.")

    p_count = count_matching(df, "P")
    ps_count = count_matching(df, "PS")

    print(f"Records in class P:     {p_count:,}")
    print(f"Records in subclass PS: {ps_count:,}")


if __name__ == "__main__":
    main()
