"""
Report match-rate diagnostics for the GeoNames direct-matching pathway
(`imprints.geonames_geocode`): validate it against the existing Nominatim
results, report how much of the corpus it resolves without touching the
LLM+Nominatim pipeline at all, and confirm the athens/columbia disambiguation
cases resolve correctly.

Three independent reports, each printable on its own:

- `--llm_validation`: joins `imprints.geonames_geocode`'s `llm` mode output
  (matching the *existing* `llm_normalized_place` strings against GeoNames)
  to the existing `data/PS/llm_geocode_nominatim.csv` on `places_clean`, and
  reports how often the two agree on country code and how close their
  coordinates are. Needs no new OpenAI or Nominatim calls.
- `--direct_coverage`: from `imprints.geonames_geocode`'s `direct` mode
  output, reports what fraction of geo_key groups/records the new pathway
  resolves without the LLM, broken out by whether place_name_008 was
  present.
- `--disambiguation`: prints the athens/{Georgia,Ohio,Illinois} and
  columbia/{Missouri,South Carolina} groups' resolved places side by side,
  from the direct-mode output -- the same cases used to verify the 008 key
  disambiguation work itself (see imprints.place_keys).

All three run by default; pass any of the flags to run only that subset.

Usage:
    python -m imprints.geocode_compare \\
        --geonames_llm_csv data/PS/geonames_llm.csv \\
        --llm_nominatim_csv data/PS/llm_geocode_nominatim.csv \\
        --geonames_direct_csv data/PS/geonames_direct.csv
"""

import argparse
import math

import pandas as pd

EARTH_RADIUS_KM = 6371.0
AGREEMENT_DISTANCE_KM = 50.0

DISAMBIGUATION_CASES = [
    ("athens", ["Georgia", "Ohio", "Illinois"]),
    ("columbia", ["Missouri", "South Carolina"]),
]


def _haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _pct(n, total):
    return f"{n:,} ({n / total:.1%})" if total else f"{n:,} (n/a)"


def print_llm_validation_report(geonames_llm_csv, llm_nominatim_csv):
    print("=" * 72)
    print("VALIDATION: GeoNames-matched LLM strings vs. existing Nominatim results")
    print("=" * 72)

    geonames = pd.read_csv(geonames_llm_csv)
    nominatim = pd.read_csv(
        llm_nominatim_csv,
        usecols=[
            "places_clean",
            "llm_nominatim_found",
            "llm_nominatim_country_code",
            "llm_nominatim_lat",
            "llm_nominatim_lon",
        ],
    )
    merged = geonames.merge(nominatim, on="places_clean", how="left")

    total = len(merged)
    matched = merged["geonames_matched"]
    print(f"Rows: {total:,}")
    print(f"GeoNames matched: {_pct(int(matched.sum()), total)}")

    both = matched & merged["llm_nominatim_found"].fillna(False)
    print(f"Both GeoNames and Nominatim resolved: {_pct(int(both.sum()), total)}")

    if both.sum():
        sub = merged[both].copy()
        country_agree = (
            sub["geonames_country_code"].str.lower()
            == sub["llm_nominatim_country_code"].str.lower()
        )
        sub["distance_km"] = [
            _haversine_km(a, b, c, d)
            for a, b, c, d in zip(
                sub["geonames_lat"],
                sub["geonames_lon"],
                sub["llm_nominatim_lat"],
                sub["llm_nominatim_lon"],
            )
        ]
        close = sub["distance_km"].notna() & (
            sub["distance_km"] <= AGREEMENT_DISTANCE_KM
        )
        agree = country_agree & close
        print(f"  Country code agrees: {_pct(int(country_agree.sum()), len(sub))}")
        print(
            f"  Within {AGREEMENT_DISTANCE_KM:.0f}km: {_pct(int(close.sum()), len(sub))}"
        )
        print(f"  Both (agree): {_pct(int(agree.sum()), len(sub))}")

        disagreements = sub[~agree].sort_values("n_records", ascending=False)
        if len(disagreements):
            n_shown = min(10, len(disagreements))
            print(f"\n  Sample disagreements (top {n_shown} by n_records):")
            for _, row in disagreements.head(n_shown).iterrows():
                print(
                    f"    {row['places_clean']!r} -> llm={row['llm_normalized_place']!r} "
                    f"geonames=({row['geonames_name']}, {row['geonames_country_code']}) "
                    f"nominatim_country={row['llm_nominatim_country_code']} "
                    f"distance_km={row['distance_km']} n_records={row['n_records']}"
                )
    print()


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
    parser.add_argument("--geonames_llm_csv", default="data/PS/geonames_llm.csv")
    parser.add_argument(
        "--llm_nominatim_csv", default="data/PS/llm_geocode_nominatim.csv"
    )
    parser.add_argument("--geonames_direct_csv", default="data/PS/geonames_direct.csv")
    parser.add_argument(
        "--llm_validation",
        action="store_true",
        help="Run only the LLM-vs-Nominatim validation report.",
    )
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

    run_all = not (args.llm_validation or args.direct_coverage or args.disambiguation)

    if run_all or args.llm_validation:
        print_llm_validation_report(args.geonames_llm_csv, args.llm_nominatim_csv)
    if run_all or args.direct_coverage:
        print_direct_coverage_report(args.geonames_direct_csv)
    if run_all or args.disambiguation:
        print_disambiguation_samples(args.geonames_direct_csv)


if __name__ == "__main__":
    main()
