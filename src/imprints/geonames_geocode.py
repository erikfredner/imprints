"""
Geocode places of publication directly against a local GeoNames gazetteer,
using the MARC 008 place-of-publication signal to scope the lookup -- a
free, deterministic, entirely local geocoder for the ~94% of records that
carry a `place_name_008` value.

Two modes, selected by the `mode` positional argument:

- `direct` (default): groups `data/PS/data.csv` by `geo_key`
  (`imprints.place_keys.build_geo_key`, reusing
  `imprints.place_keys.build_places` for the grouping itself), resolves
  each group's `place_name_008` to a GeoNames scope via
  `imprints.marc_place_geonames.resolve_geo_scope`, and matches
  `places_clean` against the gazetteer within that scope. This is the
  primary geocoding pass: whatever it resolves (`geonames_matched=True`)
  never needs to go through the LLM normalization pass at all.
- `llm`: matches `imprints.llm_geocode`'s already-produced
  `llm_normalized_place` strings (`"city, state"` / `"city, country"` /
  `"city, state, country"`, see that module's prompt) against the same
  gazetteer, resolving the trailing state/country segment(s) to a scope via
  `imprints.marc_place_geonames.resolve_scope_by_name`. This is the second
  half of the fallback for whatever `direct` mode couldn't resolve -- run
  `imprints.llm_geocode` against just that residual to clean up typos/noise
  the gazetteer lookup can't, then geocode the results here. `geo_key` is
  passed through from the input so the result can be merged back onto
  `data.csv`/`direct` mode's output via `imprints.join_geocoded`.

Matching (`match_place`) is scope-restricted exact-name lookup against
GeoNames populated-place (feature class `P`) rows, normalized with
`imprints.data_cleaning.clean_string` -- the same normalization already
applied to `places_clean`, so the two compare equal. If that fails,
`imprints.place_canonicalization.canonicalize_place` is tried as a second
pass, to strip a redundant trailing state token (e.g. "boston mass" ->
"boston") before retrying. Multiple equally-named candidates within a scope
are resolved by highest population, flagged `geonames_ambiguous=True` with
the candidate count for QA visibility, rather than left unmatched -- a
places_clean/LLM string with no candidate at all is left unmatched.

Requires GeoNames per-country dumps (see `imprints.marc_place_geonames`'
docstring for the download commands) for whichever countries
`imprints.marc_place_008_geonames.csv` resolves to -- currently US, CA, GB,
AU.

Usage:
    python -m imprints.geonames_geocode \\
        --input_csv data/PS/data.csv \\
        --output_csv data/PS/geonames_direct.csv

    python -m imprints.geonames_geocode llm \\
        --input_csv data/PS/llm_geocode.csv \\
        --output_csv data/PS/geonames_llm.csv
"""

import argparse
import csv
from collections import defaultdict

import pandas as pd

from imprints import marc_place_geonames, place_keys
from imprints.data_cleaning import clean_string
from imprints.place_canonicalization import canonicalize_place

# US/CA/GB/AU are the only countries place_name_008 ever names (mode="direct"
# only needs these four). Everything else is for mode="llm": countries named
# by imprints.llm_geocode's normalized answers for records place_name_008
# can't scope -- see the country list's provenance/threshold note at
# imprints.marc_place_geonames._COUNTRY_NAME_TO_CODE_CI.
DEFAULT_COUNTRY_FILES = [
    "data/geonames/US.txt",
    "data/geonames/CA.txt",
    "data/geonames/GB.txt",
    "data/geonames/AU.txt",
    "data/geonames/DE.txt",
    "data/geonames/IT.txt",
    "data/geonames/FR.txt",
    "data/geonames/IE.txt",
    "data/geonames/IN.txt",
    "data/geonames/JP.txt",
    "data/geonames/ES.txt",
    "data/geonames/CN.txt",
    "data/geonames/CH.txt",
    "data/geonames/RU.txt",
    "data/geonames/MX.txt",
    "data/geonames/NL.txt",
    "data/geonames/SE.txt",
    "data/geonames/PL.txt",
    "data/geonames/FI.txt",
    "data/geonames/RO.txt",
    "data/geonames/BR.txt",
    "data/geonames/IL.txt",
    "data/geonames/DK.txt",
    "data/geonames/CU.txt",
    "data/geonames/ZA.txt",
    "data/geonames/NG.txt",
    "data/geonames/BE.txt",
    "data/geonames/AR.txt",
    "data/geonames/PR.txt",
    "data/geonames/TW.txt",
    "data/geonames/UA.txt",
    "data/geonames/AT.txt",
    "data/geonames/NZ.txt",
    "data/geonames/LB.txt",
    "data/geonames/KR.txt",
    "data/geonames/CZ.txt",
    "data/geonames/NO.txt",
    "data/geonames/PT.txt",
    "data/geonames/PH.txt",
    "data/geonames/GR.txt",
    "data/geonames/VI.txt",
    "data/geonames/EG.txt",
    "data/geonames/TR.txt",
    "data/geonames/EC.txt",
    "data/geonames/IR.txt",
    "data/geonames/BA.txt",
    "data/geonames/CL.txt",
]
DEFAULT_ADMIN1_CODES = "data/geonames/admin1CodesASCII.txt"
DEFAULT_COUNTRIES = [
    "US",
    "CA",
    "GB",
    "AU",
    "DE",
    "IT",
    "FR",
    "IE",
    "IN",
    "JP",
    "ES",
    "CN",
    "CH",
    "RU",
    "MX",
    "NL",
    "SE",
    "PL",
    "FI",
    "RO",
    "BR",
    "IL",
    "DK",
    "CU",
    "ZA",
    "NG",
    "BE",
    "AR",
    "PR",
    "TW",
    "UA",
    "AT",
    "NZ",
    "LB",
    "KR",
    "CZ",
    "NO",
    "PT",
    "PH",
    "GR",
    "VI",
    "EG",
    "TR",
    "EC",
    "IR",
    "BA",
    "CL",
]

FEATURE_CLASS_POPULATED_PLACE = "P"
# 0-indexed column positions in a GeoNames main data file (19 tab-separated
# columns; see https://download.geonames.org/export/dump/readme.txt).
COL_GEONAMEID = 0
COL_NAME = 1
COL_ASCIINAME = 2
COL_ALTERNATENAMES = 3
COL_LATITUDE = 4
COL_LONGITUDE = 5
COL_FEATURE_CLASS = 6
COL_COUNTRY_CODE = 8
COL_ADMIN1_CODE = 10
COL_POPULATION = 14

DIRECT_OUTPUT_FIELDS = [
    "geo_key",
    "places_clean",
    "place_name_008",
    "geonames_matched",
    "geonames_id",
    "geonames_name",
    "geonames_country_code",
    "geonames_admin1_code",
    "geonames_lat",
    "geonames_lon",
    "geonames_population",
    "geonames_ambiguous",
    "n_records",
]
LLM_OUTPUT_FIELDS = [
    "geo_key",
    "places_clean",
    "llm_normalized_place",
    "geonames_matched",
    "geonames_id",
    "geonames_name",
    "geonames_country_code",
    "geonames_admin1_code",
    "geonames_lat",
    "geonames_lon",
    "geonames_population",
    "geonames_ambiguous",
    "n_records",
]


def load_geonames_index(paths):
    """Parse GeoNames per-country dump files into two indices, each shaped
    {(country_code, normalized_name): [row, ...]} and keeping only populated
    places (feature class 'P'): `primary` (keyed by `name`/`asciiname`) and
    `alternate` (keyed by every entry in `alternatenames` -- this main-dump
    column, not the separate alternateNamesV2 dataset -- already downloaded,
    no extra fetch needed). All keys are normalized with
    `imprints.data_cleaning.clean_string` so they compare equal to
    `places_clean`.

    Kept as two tiers, not one merged index, because `alternatenames` mixes
    a place's genuinely common short forms with foreign-language names and
    stale historical ones -- e.g. Lemont, IL's alternate names include
    "Athens" (a former name, population 16,788), which would otherwise
    outrank the real Athens, IL (population 1,938) on a population tie-break
    if both tiers were merged. `match_place` only consults `alternate` when
    `primary` has no candidate in scope. Indexing alternate names at all
    still matters for the single most common place in this corpus: GeoNames'
    own `name`/`asciiname` for New York City is literally "New York City" --
    the bare "New York" catalogers almost always use is only present in
    `alternatenames`, and (scoped to NY state) no other populated place's
    primary name collides with it. Non-Latin-script alternates clean to an
    empty string and are dropped automatically."""
    primary = defaultdict(list)
    alternate = defaultdict(list)
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if fields[COL_FEATURE_CLASS] != FEATURE_CLASS_POPULATED_PLACE:
                    continue
                country_code = fields[COL_COUNTRY_CODE]
                population_raw = fields[COL_POPULATION]
                row = {
                    "geonameid": fields[COL_GEONAMEID],
                    "name": fields[COL_NAME],
                    "admin1_code": fields[COL_ADMIN1_CODE] or None,
                    "lat": float(fields[COL_LATITUDE]),
                    "lon": float(fields[COL_LONGITUDE]),
                    "population": int(population_raw) if population_raw else 0,
                }
                primary_keys = {
                    clean_string(fields[COL_NAME]),
                    clean_string(fields[COL_ASCIINAME]),
                }
                for key in primary_keys:
                    if key:
                        primary[(country_code, key)].append(row)
                if fields[COL_ALTERNATENAMES]:
                    alt_keys = {
                        clean_string(n) for n in fields[COL_ALTERNATENAMES].split(",")
                    } - primary_keys
                    for key in alt_keys:
                        if key:
                            alternate[(country_code, key)].append(row)
    return {"primary": primary, "alternate": alternate}


def _rank(candidates, country_code, admin1_code):
    if admin1_code:
        candidates = [c for c in candidates if c["admin1_code"] == admin1_code]
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda c: c["population"], reverse=True)
    best = candidates[0]
    return {
        "geonames_id": best["geonameid"],
        "geonames_name": best["name"],
        "geonames_country_code": country_code,
        "geonames_admin1_code": best["admin1_code"],
        "geonames_lat": best["lat"],
        "geonames_lon": best["lon"],
        "geonames_population": best["population"],
        "geonames_ambiguous": len(candidates) > 1,
    }


def _best_candidate(name, country_code, admin1_code, index):
    """Look up `name` in `index["primary"]`, falling back to
    `index["alternate"]` only if the primary tier has no candidate in scope
    -- see load_geonames_index for why the tiers aren't merged."""
    result = _rank(
        index["primary"].get((country_code, name), []), country_code, admin1_code
    )
    if result is not None:
        return result
    return _rank(
        index["alternate"].get((country_code, name), []), country_code, admin1_code
    )


def match_place(name, scope, index):
    """Match `name` (a places_clean-normalized string) against the
    populated-place `index` within `scope` (country_code, admin1_code_or_
    None). Tries an exact match first, then -- if that fails -- retries
    against `canonicalize_place(name)`'s city-only portion (dropping a
    trailing state token, e.g. "waterville me" -> "waterville"). The retry
    always runs when canonicalize_place finds a city+state split, even if
    its output is textually identical to `name` -- `name` is frequently
    already in "<city> <postal code>" form (e.g. "waterville me"), which
    GeoNames indexes as bare "Waterville", not "waterville me". Returns a
    geonames_* field dict, or None if nothing matches at either step."""
    if not name:
        return None
    country_code, admin1_code = scope

    result = _best_candidate(name, country_code, admin1_code, index)
    if result is not None:
        return result

    canonical = canonicalize_place(name)
    if canonical:
        tokens = canonical.split()
        if len(tokens) > 1:
            city_part = " ".join(tokens[:-1])
            if city_part != name:
                result = _best_candidate(city_part, country_code, admin1_code, index)
    return result


def _row_common_fields(result):
    if result is None:
        return {
            "geonames_matched": False,
            "geonames_id": None,
            "geonames_name": None,
            "geonames_country_code": None,
            "geonames_admin1_code": None,
            "geonames_lat": None,
            "geonames_lon": None,
            "geonames_population": None,
            "geonames_ambiguous": None,
        }
    return {"geonames_matched": True, **result}


def run_direct(input_csv, crosswalk_path, index, output_csv):
    print(f"Loading {input_csv}")
    wanted = ["places", "places_clean", "place_name_008", "place_752"]
    header = pd.read_csv(input_csv, nrows=0)
    usecols = [c for c in wanted if c in header.columns]
    df = pd.read_csv(input_csv, usecols=usecols)
    groups = place_keys.build_places(df)
    print(f"{len(groups)} unique (places_clean, place_name_008) groups.")

    n_matched = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DIRECT_OUTPUT_FIELDS)
        writer.writeheader()
        for _, row in groups.iterrows():
            scope = marc_place_geonames.resolve_geo_scope(
                row["place_name_008"], crosswalk_path
            )
            result = match_place(row["places_clean"], scope, index) if scope else None
            if result is not None:
                n_matched += 1
            writer.writerow(
                {
                    "geo_key": row["geo_key"],
                    "places_clean": row["places_clean"],
                    "place_name_008": row["place_name_008"],
                    "n_records": row["n_records"],
                    **_row_common_fields(result),
                }
            )

    print(
        f"Matched {n_matched}/{len(groups)} groups "
        f"({n_matched / len(groups):.1%}). Wrote {output_csv}"
    )


def _resolve_llm_scope(llm_normalized_place, admin1_names, countries):
    """Resolve an llm_normalized_place string ("city, state" / "city,
    country" / "city, state, country") to (scope, city). Tries the
    rightmost segment first; an admin1-level match is taken immediately
    (most specific), a country-level match is kept only if no segment
    yields an admin1-level match. Returns (None, city) if no segment
    resolves."""
    parts = [p.strip() for p in str(llm_normalized_place).split(",") if p.strip()]
    if len(parts) < 2:
        return None, None
    city = parts[0]

    best_scope = None
    for segment in reversed(parts[1:]):
        try:
            scope = marc_place_geonames.resolve_scope_by_name(
                segment, admin1_names, countries
            )
        except ValueError:
            continue
        if scope is None:
            continue
        if scope[1] is not None:
            return scope, city
        if best_scope is None:
            best_scope = scope
    return best_scope, city


def run_llm(input_csv, admin1_codes_path, countries, index, output_csv):
    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)
    admin1_names = marc_place_geonames.load_admin1_names(admin1_codes_path, countries)

    n_matched = 0
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LLM_OUTPUT_FIELDS)
        writer.writeheader()
        for _, row in df.iterrows():
            llm_place = row.get("llm_normalized_place")
            result = None
            if pd.notna(llm_place) and str(llm_place).strip():
                scope, city = _resolve_llm_scope(llm_place, admin1_names, countries)
                if scope is not None:
                    result = match_place(clean_string(city), scope, index)
            if result is not None:
                n_matched += 1
            writer.writerow(
                {
                    "geo_key": row.get("geo_key"),
                    "places_clean": row["places_clean"],
                    "llm_normalized_place": llm_place,
                    "n_records": row["n_records"],
                    **_row_common_fields(result),
                }
            )

    print(
        f"Matched {n_matched}/{len(df)} rows ({n_matched / len(df):.1%}). "
        f"Wrote {output_csv}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Geocode places of publication directly against a local "
        "GeoNames gazetteer, scoped by the MARC 008 place-of-publication "
        "signal."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["direct", "llm"],
        default="direct",
        help="'direct' (default) matches data.csv's places_clean/"
        "place_name_008 groups. 'llm' matches imprints.llm_geocode's "
        "llm_normalized_place strings instead -- the second stage of the "
        "fallback for whatever 'direct' couldn't resolve.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Default: data/PS/data.csv for mode=direct, "
        "data/PS/llm_geocode.csv for mode=llm.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Default: data/PS/geonames_direct.csv for mode=direct, "
        "data/PS/geonames_llm.csv for mode=llm.",
    )
    parser.add_argument(
        "--country_files",
        default=None,
        help="Comma-separated paths to GeoNames per-country dump files. "
        f"Default: {','.join(DEFAULT_COUNTRY_FILES)}",
    )
    parser.add_argument("--admin1_codes", default=DEFAULT_ADMIN1_CODES)
    parser.add_argument(
        "--crosswalk_csv",
        default=marc_place_geonames.CROSSWALK_PATH,
        help="Path to the frozen place_name_008 -> GeoNames scope crosswalk "
        "(mode=direct only). See imprints.marc_place_geonames.",
    )
    parser.add_argument(
        "--countries",
        default=",".join(DEFAULT_COUNTRIES),
        help="Comma-separated GeoNames country codes available for scope "
        "resolution (mode=llm only; mode=direct uses whatever the frozen "
        "crosswalk already resolved).",
    )
    args = parser.parse_args()

    country_files = (
        args.country_files.split(",") if args.country_files else DEFAULT_COUNTRY_FILES
    )
    print(f"Loading GeoNames index from {country_files}")
    index = load_geonames_index(country_files)
    n_primary = sum(len(v) for v in index["primary"].values())
    n_alternate = sum(len(v) for v in index["alternate"].values())
    print(f"Indexed {n_primary} primary-name and {n_alternate} alternate-name entries.")

    if args.mode == "direct":
        input_csv = args.input_csv or "data/PS/data.csv"
        output_csv = args.output_csv or "data/PS/geonames_direct.csv"
        run_direct(input_csv, args.crosswalk_csv, index, output_csv)
    else:
        input_csv = args.input_csv or "data/PS/llm_geocode.csv"
        output_csv = args.output_csv or "data/PS/geonames_llm.csv"
        countries = [c.strip() for c in args.countries.split(",") if c.strip()]
        run_llm(input_csv, args.admin1_codes, countries, index, output_csv)


if __name__ == "__main__":
    main()
