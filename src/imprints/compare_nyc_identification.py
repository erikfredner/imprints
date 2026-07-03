"""
Compare the existing NYC identification (from `260`/`264` `$a` place text,
via `imprints.data_cleaning.get_target_cities` matching against
`nyc_variants.txt`) to a new NYC label built from `752`'s hierarchical city
subfield (`$d`) -- the only newly captured MARC field (008 place code, 044
country codes, 752 hierarchical place) with *city*-level granularity. 008
bytes 15-17 and 044 codes are state/country level (e.g. `"nyu"` = New York
State) and can't distinguish New York City from Buffalo or Albany, so they
are not used to build the new label; their presence is only reported as
diagnostic context, cross-tabulated against both labels.

Operates at the *record* level, before `places`/752 occurrences are
exploded: a record counts as NYC under a system if ANY of its components do
under that system. This keeps the comparison apples-to-apples with the old
system, which is inherently record-level before `data_cleaning.
cleaning_pipeline` explodes `places` into one row per component city.
Reuses `get_target_cities`/`expand_places` for both signals, so the only
difference between "old" and "new" is the source subfield, not the matching
logic.

Requires pickles produced by a `data_collection` run *after* 008/044/752
capture was added (see README.md) -- older pickles won't have
`country_codes_044`/`place_hierarchy_752` and this script will refuse to
run against them.

Usage:
    python -m imprints.compare_nyc_identification \\
        --input_dir data/PS --class_range PS
"""

import argparse

from imprints.data_cleaning import (
    _parse_place_code_008,
    expand_places,
    filter_classifications,
    get_target_cities,
    load_pickles_to_dataframe,
    parse_range_spec,
)

REQUIRED_NEW_FIELDS = ("country_codes_044", "place_hierarchy_752")


def record_old_nyc(places):
    """True if any 260/264-derived place component resolves to NYC."""
    return any(get_target_cities(p) == "New York City" for p in expand_places(places))


def place_752_cities(place_hierarchy_752):
    """Each 752 occurrence's $d (city) subfield text, if present."""
    if not place_hierarchy_752:
        return []
    cities = []
    for occ in place_hierarchy_752:
        city = dict(occ.get("subfields", [])).get("d")
        if city:
            cities.append(city)
    return cities


def record_new_nyc(place_hierarchy_752):
    """True if any 752 $d (city) subfield resolves to NYC."""
    return any(
        get_target_cities(c) == "New York City"
        for c in place_752_cities(place_hierarchy_752)
    )


def record_has_752_city(place_hierarchy_752):
    """True if the record has at least one 752 occurrence with a $d city,
    regardless of what it resolves to -- distinguishes "752 said not-NYC"
    from "752 has no city data at all" when reading new_nyc == False."""
    return len(place_752_cities(place_hierarchy_752)) > 0


def record_ny_state_signal(field_008, country_codes_044):
    """Weak corroborating signal only: True if the 008 place code or any 044
    country code is 'nyu' (MARC code for New York State). Country/state
    granularity -- NOT used to build the new NYC label, since it can't
    distinguish NYC from the rest of the state. Reported separately as
    diagnostic context."""
    codes = list(country_codes_044 or [])
    place_code_008 = _parse_place_code_008(field_008)
    if place_code_008:
        codes.append(place_code_008)
    return "nyu" in codes


def build_comparison(df, class_range):
    """Filter to class_range and compute old_nyc/new_nyc/diagnostic columns
    on the record-level (pre-explode) DataFrame. Raises SystemExit if the
    input lacks the new MARC fields."""
    missing = [c for c in REQUIRED_NEW_FIELDS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Input is missing {missing} -- these pickles predate 008/044/752 "
            "capture. Re-run `python -m imprints.data_collection` against the "
            "raw MARC .xml.gz files to regenerate them before running this "
            "comparison."
        )

    df = df.copy()
    df["matching_classifications"] = df["classifications"].map(
        lambda clist: filter_classifications(clist, class_range)
    )
    df = df[df["matching_classifications"].apply(lambda m: len(m) > 0)]

    df["old_nyc"] = df["places"].map(record_old_nyc)
    df["new_nyc"] = df["place_hierarchy_752"].map(record_new_nyc)
    df["has_752_city"] = df["place_hierarchy_752"].map(record_has_752_city)
    df["place_code_008"] = df["field_008"].map(_parse_place_code_008)
    df["ny_state_signal"] = df.apply(
        lambda r: record_ny_state_signal(
            r.get("field_008"), r.get("country_codes_044")
        ),
        axis=1,
    )
    return df


def _pct(n, total):
    return f"{n} ({n / total:.1%})" if total else f"{n} (n/a)"


def print_summary(df, sample_size):
    total = len(df)
    old_nyc = df["old_nyc"]
    new_nyc = df["new_nyc"]
    has_752 = df["has_752_city"]
    ny_state = df["ny_state_signal"]

    both_nyc = (old_nyc & new_nyc).sum()
    both_other = (~old_nyc & ~new_nyc).sum()
    old_only = old_nyc & ~new_nyc
    new_only = ~old_nyc & new_nyc

    print(f"Total records evaluated: {total:,}")
    print(f"  Old system (260/264 $a) NYC:      {_pct(int(old_nyc.sum()), total)}")
    print(f"  New system (752 $d city) NYC:     {_pct(int(new_nyc.sum()), total)}")
    print(
        f"  Records with any 752 $d city data: {_pct(int(has_752.sum()), total)} "
        "(coverage -- new_nyc=False mostly means 'no 752 data', not 'resolved "
        "to non-NYC')"
    )
    print()

    print("Agreement:")
    print(f"  Both say NYC:              {_pct(int(both_nyc), total)}")
    print(f"  Both say Not-NYC:          {_pct(int(both_other), total)}")
    print(f"  Total agreement:           {_pct(int(both_nyc + both_other), total)}")
    print()

    print("Disagreement:")
    print(f"  Old says NYC, new does not: {_pct(int(old_only.sum()), total)}")
    old_only_no_752 = old_only & ~has_752
    old_only_other_752 = old_only & has_752
    print(
        f"    - new has no 752 city data at all:        "
        f"{_pct(int(old_only_no_752.sum()), int(old_only.sum()) or 1)}"
    )
    print(
        f"    - new has 752 city data resolving elsewhere: "
        f"{_pct(int(old_only_other_752.sum()), int(old_only.sum()) or 1)}"
    )
    print(f"  New says NYC, old does not: {_pct(int(new_only.sum()), total)}")
    print()

    print(
        "Diagnostic: 008/044 New York State ('nyu') signal (not part of either label):"
    )
    print(
        f"  Records with a 'nyu' 008/044 code:        {_pct(int(ny_state.sum()), total)}"
    )
    print(
        f"    - also old_nyc:  {_pct(int((ny_state & old_nyc).sum()), int(ny_state.sum()) or 1)}"
    )
    print(
        f"    - also new_nyc:  {_pct(int((ny_state & new_nyc).sum()), int(ny_state.sum()) or 1)}"
    )
    print(
        f"    - neither (state=NY but no city-level NYC signal from either system): "
        f"{_pct(int((ny_state & ~old_nyc & ~new_nyc).sum()), int(ny_state.sum()) or 1)}"
    )
    print()

    if sample_size:
        print(f"Sample of up to {sample_size} 'old says NYC, new does not' records:")
        for _, row in df[old_only].head(sample_size).iterrows():
            print(
                f"  places={row['places']!r}  752_cities={place_752_cities(row['place_hierarchy_752'])!r}"
            )
        print()
        print(f"Sample of up to {sample_size} 'new says NYC, old does not' records:")
        for _, row in df[new_only].head(sample_size).iterrows():
            print(
                f"  places={row['places']!r}  752_cities={place_752_cities(row['place_hierarchy_752'])!r}"
            )


def export_ny_state_non_nyc(df, output_csv):
    """Write the diagnostic "state=NY but no city-level NYC signal from
    either system" group (ny_state_signal & ~old_nyc & ~new_nyc) to CSV --
    plausibly Buffalo/Albany/Ithaca-type non-NYC New York State publishers,
    though some may be old-system misses worth a closer look. One row per
    record; `places` (the raw 260/264 $a open-string text) is joined with
    "; " since a record can list more than one place."""
    mask = df["ny_state_signal"] & ~df["old_nyc"] & ~df["new_nyc"]
    out = df[mask][
        ["lccn", "title", "places", "place_code_008", "country_codes_044"]
    ].copy()
    out["places"] = out["places"].map(lambda p: "; ".join(p or []))
    out["country_codes_044"] = out["country_codes_044"].map(
        lambda c: "; ".join(c or [])
    )
    out.to_csv(output_csv, index=False)
    return len(out)


def main():
    parser = argparse.ArgumentParser(
        description="Compare the old (260/264 $a) NYC identification to a "
        "new one built from 752's $d city subfield."
    )
    parser.add_argument(
        "--input_dir", default="data/PS", help="Directory with .pkl files"
    )
    parser.add_argument("--class_range", default="PS", help="e.g. PS or PR9000-PR9999")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of example mismatched records to print per direction (0 to disable)",
    )
    parser.add_argument(
        "--export_ny_state_non_nyc_csv",
        default=None,
        help="If given, write the 'NY state code but no city-level NYC signal "
        "from either system' record group (place text + context) to this CSV path.",
    )
    args = parser.parse_args()

    print(f"Loading pickles from {args.input_dir}")
    df = load_pickles_to_dataframe(args.input_dir)
    print(f"Loaded {len(df):,} records.")

    class_range = parse_range_spec(args.class_range)
    df = build_comparison(df, class_range)
    print_summary(df, args.sample_size)

    if args.export_ny_state_non_nyc_csv:
        n = export_ny_state_non_nyc(df, args.export_ny_state_non_nyc_csv)
        print(f"\nWrote {n:,} rows to {args.export_ny_state_non_nyc_csv}")


if __name__ == "__main__":
    main()
