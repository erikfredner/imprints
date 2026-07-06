"""
Report match-rate diagnostics for the GeoNames direct-matching pathway
(`imprints.geonames_geocode`): report how much of the corpus it resolves
without needing the LLM-normalization residual pass at all, and confirm the
athens/columbia disambiguation cases resolve correctly.

Two independent reports, each printable on its own:

- `--direct_coverage`: from `imprints.geonames_geocode`'s `direct` mode
  output, reports what fraction of geo_key groups/records the pathway
  resolves without the LLM, broken out by whether place_name_008 was
  present.
- `--disambiguation`: prints the athens/{Georgia,Ohio,Illinois} and
  columbia/{Missouri,South Carolina} groups' resolved places side by side,
  from the direct-mode output -- the same cases used to verify the 008 key
  disambiguation work itself (see imprints.place_keys).

Both run by default; pass either flag to run only that one.

Usage:
    python -m imprints.geocode_compare \\
        --geonames_direct_csv data/PS/geonames_direct.csv
"""

import argparse

import pandas as pd

DISAMBIGUATION_CASES = [
    ("athens", ["Georgia", "Ohio", "Illinois"]),
    ("columbia", ["Missouri", "South Carolina"]),
]


def _pct(n, total):
    return f"{n:,} ({n / total:.1%})" if total else f"{n:,} (n/a)"


def print_direct_coverage_report(geonames_direct_csv):
    print("=" * 72)
    print("COVERAGE: direct GeoNames matching (imprints.geonames_geocode direct)")
    print("=" * 72)

    df = pd.read_csv(geonames_direct_csv)
    total_groups = len(df)
    total_records = int(df["n_records"].sum())
    matched = df["geonames_matched"]

    print(f"Groups: {total_groups:,}  Records: {total_records:,}")
    print(f"Matched groups: {_pct(int(matched.sum()), total_groups)}")
    print(
        f"Matched records: {_pct(int(df.loc[matched, 'n_records'].sum()), total_records)}"
    )

    has_008 = df["place_name_008"].notna()
    for label, mask in [
        ("with place_name_008", has_008),
        ("without place_name_008", ~has_008),
    ]:
        sub = df[mask]
        if not len(sub):
            continue
        sub_records = int(sub["n_records"].sum())
        sub_matched_records = int(sub.loc[sub["geonames_matched"], "n_records"].sum())
        print(
            f"  Records {label}: {sub_records:,}, matched: {_pct(sub_matched_records, sub_records)}"
        )

    ambiguous = df["geonames_ambiguous"].fillna(False).astype(bool)
    print(
        f"Ambiguous matches (population tie-break used): "
        f"{_pct(int(ambiguous.sum()), int(matched.sum()) or 1)} of matched groups, "
        f"{int(df.loc[ambiguous, 'n_records'].sum()):,} records"
    )
    print()


def print_disambiguation_samples(geonames_direct_csv, cases=DISAMBIGUATION_CASES):
    print("=" * 72)
    print("DISAMBIGUATION CHECK: same bare city name, different place_name_008")
    print("=" * 72)

    df = pd.read_csv(geonames_direct_csv)
    cols = [
        "places_clean",
        "place_name_008",
        "geonames_matched",
        "geonames_name",
        "geonames_admin1_code",
        "geonames_lat",
        "geonames_lon",
        "geonames_ambiguous",
        "n_records",
    ]
    for places_clean, states in cases:
        sub = df[
            (df["places_clean"] == places_clean) & (df["place_name_008"].isin(states))
        ]
        print(f"\n{places_clean!r} x {states}:")
        print(sub[cols].to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Report match-rate diagnostics for the GeoNames direct "
        "geocoding pathway."
    )
    parser.add_argument("--geonames_direct_csv", default="data/PS/geonames_direct.csv")
    parser.add_argument(
        "--direct_coverage",
        action="store_true",
        help="Run only the direct-pathway coverage report.",
    )
    parser.add_argument(
        "--disambiguation",
        action="store_true",
        help="Run only the disambiguation check.",
    )
    args = parser.parse_args()

    run_all = not (args.direct_coverage or args.disambiguation)

    if run_all or args.direct_coverage:
        print_direct_coverage_report(args.geonames_direct_csv)
    if run_all or args.disambiguation:
        print_disambiguation_samples(args.geonames_direct_csv)


if __name__ == "__main__":
    main()
