"""Build NYC-vs-other imprint count tables for *every* LC range.

The essay shows that for class PS the New-York-City share of imprints rises
through ~1958 then falls. To ask whether that pattern is particular to PS,
general to LC, or shared by a subset of ranges, we need the same NYC-share
series for every LC range.

The ``data/PS`` pickles retain *all* records from the raw MARC XML (the
``matches_class_range`` flag is just a boolean), so we can recover every range
by streaming those pickles once -- no re-parse of the raw XML. This module
streams them **one file at a time** and accumulates small per-(range, year,
city) count tables, applying the same normalization as
``imprints.data_cleaning`` (reused, not duplicated).

Assignment is by *membership*: a row counts toward every range any of its
classifications match (matching the first-matching semantics of the existing
PS pipeline). "No place of publication" rows are dropped so the share compares
like with like across ranges.

Outputs (to ``--output_dir``), each with columns ``key,year_min,nyc,other``:

* ``counts_letter.csv``   -- top-level class (first letter, e.g. ``P``)
* ``counts_subclass.csv`` -- subclass (full alpha prefix, e.g. ``PS``)
* ``counts_special.csv``  -- keys ``all`` / ``PS`` / ``not_PS``
"""

import argparse
import os
import pickle

import pandas as pd

from imprints.data_cleaning import (
    clean_string,
    expand_places,
    get_publishers_year_ints,
    get_target_cities,
    get_years_ints,
    parse_class,
)

NO_PLACE = "No place of publication"
NYC = "New York City"


def class_prefixes(classifications):
    """Return ``(letters, subclasses)`` lists of distinct LC prefixes.

    ``subclasses`` are the full alpha prefixes (e.g. ``PS``, ``PR``, ``PZ``);
    ``letters`` are their first characters (the top-level class, e.g. ``P``).
    Both are deduped so a record with two PS classifications counts once.
    """
    if not isinstance(classifications, (list, tuple)):
        classifications = [classifications] if classifications else []
    subclasses, letters = set(), set()
    for c in classifications:
        if not c:
            continue
        prefix, _ = parse_class(str(c).strip())
        if prefix:
            subclasses.add(prefix)
            letters.add(prefix[:1])
    return sorted(letters), sorted(subclasses)


def _grouped(df, key_col):
    """Counts of (key, year_min, in_nyc), exploding the list-valued key column."""
    exploded = df[["year_min", "in_nyc", key_col]].explode(key_col)
    exploded = exploded[exploded[key_col].notna()]
    g = exploded.groupby([key_col, "year_min", "in_nyc"]).size()
    g.index = g.index.set_names(["key", "year_min", "in_nyc"])
    return g


def process_file(path, start_year, end_year):
    """Normalize one pickle and return (letter, subclass, special) count Series.

    Each Series is indexed by ``(key, year_min, in_nyc)``.
    """
    with open(path, "rb") as f:
        df = pd.DataFrame(pickle.load(f))

    # year_min: earliest 4-digit year from the date or publisher subfields.
    df["year_int"] = df["year"].map(get_years_ints)
    df["pub_year_int"] = df["publishers"].map(get_publishers_year_ints)
    df["year_min"] = df[["year_int", "pub_year_int"]].min(axis=1)

    prefixes = df["classifications"].map(class_prefixes)
    df["letters"] = prefixes.map(lambda lp: lp[0])
    df["subclasses"] = prefixes.map(lambda lp: lp[1])

    df = df[df["subclasses"].map(len) > 0]
    df = df[df["year_min"].between(start_year, end_year)]

    # Explode places (compound "Boston and New York" splits first), classify NYC,
    # and drop records with no identifiable place.
    df["places"] = df["places"].map(expand_places)
    df = df.explode("places")
    df["city_group"] = df["places"].map(clean_string).map(get_target_cities)
    df = df[df["city_group"] != NO_PLACE]
    df["in_nyc"] = df["city_group"] == NYC
    df["year_min"] = df["year_min"].astype(int)

    letter = _grouped(df, "letters")
    subclass = _grouped(df, "subclasses")

    # Special partition: every row -> "all"; "PS" if it carries a PS class else "not_PS".
    has_ps = df["subclasses"].map(lambda s: "PS" in s)
    base = df.groupby(["year_min", "in_nyc"]).size()
    ps = df[has_ps].groupby(["year_min", "in_nyc"]).size()
    not_ps = df[~has_ps].groupby(["year_min", "in_nyc"]).size()
    special = pd.concat(
        {"all": base, "PS": ps, "not_PS": not_ps}, names=["key", "year_min", "in_nyc"]
    )
    return letter, subclass, special


def _to_frame(series):
    """Turn a (key, year_min, in_nyc) count Series into key,year_min,nyc,other."""
    wide = series.unstack("in_nyc", fill_value=0)
    nyc = wide[True] if True in wide.columns else 0
    other = wide[False] if False in wide.columns else 0
    out = pd.DataFrame({"nyc": nyc, "other": other}).reset_index()
    out["nyc"] = out["nyc"].astype(int)
    out["other"] = out["other"].astype(int)
    return out.sort_values(["key", "year_min"]).reset_index(drop=True)


def build(input_dir, output_dir, start_year, end_year):
    pkls = sorted(f for f in os.listdir(input_dir) if f.endswith(".pkl"))
    if not pkls:
        raise RuntimeError(f"No .pkl files found in {input_dir}")

    totals = {"letter": None, "subclass": None, "special": None}
    for i, name in enumerate(pkls, 1):
        print(f"[{i}/{len(pkls)}] {name}")
        letter, subclass, special = process_file(
            os.path.join(input_dir, name), start_year, end_year
        )
        for key, series in (
            ("letter", letter),
            ("subclass", subclass),
            ("special", special),
        ):
            totals[key] = (
                series if totals[key] is None else totals[key].add(series, fill_value=0)
            )

    os.makedirs(output_dir, exist_ok=True)
    for key in totals:
        frame = _to_frame(totals[key])
        path = os.path.join(output_dir, f"counts_{key}.csv")
        frame.to_csv(path, index=False)
        print(f"Wrote {path} ({len(frame):,} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        default="data/PS",
        help="Directory of record pickles holding ALL classes (default: data/PS)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/cross_range",
        help="Where to write the count CSVs (default: data/cross_range)",
    )
    parser.add_argument("--start-year", type=int, default=1900)
    parser.add_argument("--end-year", type=int, default=2010)
    args = parser.parse_args()

    build(args.input_dir, args.output_dir, args.start_year, args.end_year)
