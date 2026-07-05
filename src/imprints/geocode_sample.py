"""
Geocode places of publication via Nominatim.

Two modes, selected by the `mode` positional argument:

- `places` (default): spot-checks how well `places_clean` (the string
  produced by `imprints.data_cleaning.clean_string`) maps back to
  real-world places by resolving each unique value against the
  Nominatim/OpenStreetMap search API.
- `llm`: takes `imprints.llm_geocode`'s output (one row per group with an
  `llm_normalized_place` guess) and geocodes the LLM-normalized string
  instead, carrying the original `places_clean`-based Nominatim columns
  through and adding a `nominatim_match` column that flags whether the two
  Nominatim lookups (from `places_clean` vs. from `llm_normalized_place`)
  resolved to the same city/state/country.

Both modes group by `geo_key` (`imprints.place_keys.build_geo_key`), the
composite of `places_clean` and the decoded MARC 008 place-of-publication
code (`place_name_008`, ~94% prevalent in the PS range), not `places_clean`
alone. This splits ambiguous bare city names -- e.g. "Athens" is Athens
Georgia, Ohio, or Illinois depending on the record's own 008 code -- into
separate groups wherever that signal is available; places with no 008
signal key identically to `places_clean` alone, unchanged from before this
key was introduced.

Respects the Nominatim usage policy: sequential requests only, an absolute
max of 1 request/second, a custom User-Agent, and no repeated identical
queries. Because multiple `geo_key` groups can share the same `places_clean`
text (that's the point -- a "places" mode-A Nominatim lookup only ever
queries `places_clean`, not the 008 hint), `run()` caches Nominatim results
by `places_clean` text and reuses a cached result across every geo_key group
that shares it, so the number of actual API calls tracks unique
`places_clean` values, not unique groups.
See: https://operations.osmfoundation.org/policies/nominatim/

This is a long-running job (order-12k unique places at ~1 req/sec is several
hours), so it's built to resume: results are appended to --output_csv one row
at a time as they're produced, and on startup any geo_key values already
present in an existing --output_csv are skipped. If the API starts rate
limiting/blocking (403/429) or the process is interrupted, just rerun the
same command later -- it picks up where it left off.

Migrating existing results from before geo_key existed (a places_clean-only
schema, one row per unique places_clean value) so a rerun doesn't redo
~12k already-resolved places from scratch:

1. `nominatim_full.csv`: a "places" mode Nominatim query is always the raw
   `places_clean` text -- it never depends on `place_name_008` -- so old
   rows can be broadcast unchanged to every new `geo_key` group that shares
   their `places_clean` value. Build the new group table (`build_places` on
   `data/PS/data.csv`), left-merge it onto the old file on `places_clean`,
   and write the result (with `geo_key`/`place_name_008`/`place_752` added)
   as the seeded new-schema `nominatim_full.csv`. This needs zero fresh
   Nominatim calls.
2. `llm_geocode.csv` / `llm_geocode_nominatim.csv`: only groups that don't
   actually split -- their `places_clean` maps to exactly one `geo_key` in
   the new group table -- can be carried forward unchanged, since there's
   no new hint to change the LLM's answer for those. Groups that DO split
   (2+ geo_keys sharing a `places_clean`, e.g. "athens") are exactly the
   ambiguous cases this key change targets, so leave them out of the seeded
   file; the resumable "skip already-processed geo_key" logic here and in
   `imprints.llm_geocode.run()` then spends fresh API calls only on that
   (much smaller) split-group minority when you rerun the normal commands.

Usage:
    python -m imprints.geocode_sample \
        --input_csv data/PS/data.csv \
        --output_csv data/PS/nominatim_full.csv \
        --email you@example.com

    python -m imprints.geocode_sample llm \
        --input_csv data/PS/llm_geocode.csv \
        --output_csv data/PS/llm_geocode_nominatim.csv \
        --email you@example.com
"""

import argparse
import ast
import csv
import os
import time

import pandas as pd
import requests

from imprints import place_keys

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CITY_FIELDS = ("city", "town", "village", "borough", "suburb")
STATE_FIELDS = ("state", "state_district")
OUTPUT_FIELDS = [
    "geo_key",
    "places_clean",
    "place_name_008",
    "place_752",
    "places_original_example",
    "n_records",
    "nominatim_found",
    "nominatim_city",
    "nominatim_state",
    "nominatim_country",
    "nominatim_country_code",
    "nominatim_lat",
    "nominatim_lon",
    "nominatim_display_name",
]
EMPTY_NOMINATIM_RESULT = {
    "nominatim_found": False,
    "nominatim_city": None,
    "nominatim_state": None,
    "nominatim_country": None,
    "nominatim_country_code": None,
    "nominatim_lat": None,
    "nominatim_lon": None,
    "nominatim_display_name": None,
}
LLM_OUTPUT_FIELDS = [
    "geo_key",
    "places_clean",
    "place_name_008",
    "place_752",
    "places_original_example",
    "n_records",
    "nominatim_found",
    "nominatim_city",
    "nominatim_state",
    "nominatim_country",
    "llm_normalized_place",
    "llm_model",
    "llm_nominatim_found",
    "llm_nominatim_city",
    "llm_nominatim_state",
    "llm_nominatim_country",
    "llm_nominatim_country_code",
    "llm_nominatim_lat",
    "llm_nominatim_lon",
    "llm_nominatim_display_name",
    "nominatim_match",
]


def _first_non_null(series):
    """First non-null value in a groupby Series, or None if all are null."""
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else None


def _first_place_752_example(value):
    """Extract the first flattened 752 occurrence string from a place_752
    cell -- a stringified list as read back from data.csv (e.g.
    "['United States, Ohio, Athens']"), a real list, or None/NaN. Returns
    None if unparseable or empty. See imprints.data_cleaning.
    _flatten_place_hierarchy for how this column is produced."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        items = value
    else:
        try:
            items = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return None
    return items[0] if items else None


def build_places(df):
    """Group by geo_key (places_clean + place_name_008), keeping one
    representative places_clean/place_name_008/original value and a count
    per group, sorted for a deterministic, resumable order. A missing
    `place_name_008` column (older data.csv, pre-008 capture) is treated as
    all-missing, so grouping falls back to places_clean alone."""
    df = df[df["places"].notna() & df["places_clean"].notna()]
    df = df[df["places_clean"].astype(str).str.strip() != ""]
    if "place_name_008" not in df.columns:
        df = df.assign(place_name_008=None)
    if "place_752" not in df.columns:
        df = df.assign(place_752=None)

    df = df.assign(
        geo_key=[
            place_keys.build_geo_key(pc, p8)
            for pc, p8 in zip(df["places_clean"], df["place_name_008"])
        ]
    )

    grouped = (
        df.groupby("geo_key")
        .agg(
            places_clean=("places_clean", "first"),
            place_name_008=("place_name_008", "first"),
            place_752=("place_752", _first_non_null),
            places_original_example=("places", "first"),
            n_records=("places", "size"),
        )
        .reset_index()
    )
    grouped["place_752"] = grouped["place_752"].map(_first_place_752_example)
    return grouped.sort_values("geo_key").reset_index(drop=True)


def already_processed(output_csv):
    """Return the set of geo_key values already recorded in output_csv."""
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set()
    done = pd.read_csv(output_csv, usecols=["geo_key"])
    return set(done["geo_key"].astype(str))


def _seed_places_clean_cache(output_csv):
    """Build a places_clean -> Nominatim result cache from an existing
    output_csv, so a resumed run doesn't re-query Nominatim for place text
    that was already geocoded under a different geo_key in an earlier run."""
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return {}
    existing = pd.read_csv(output_csv)
    cache = {}
    for _, row in existing.iterrows():
        place = row["places_clean"]
        if place not in cache:
            cache[place] = {field: row.get(field) for field in EMPTY_NOMINATIM_RESULT}
    return cache


def geocode_place(session, place, email, user_agent, max_retries=3):
    """Query Nominatim's /search for a single place string.

    Returns a dict of result fields. Raises requests.HTTPError on a 403/429
    so the caller can stop the run rather than continue against a block.
    """
    params = {
        "q": place,
        "format": "jsonv2",
        "addressdetails": 1,
        "limit": 1,
        "email": email,
    }
    headers = {"User-Agent": user_agent}

    for attempt in range(max_retries):
        try:
            resp = session.get(
                NOMINATIM_URL, params=params, headers=headers, timeout=10
            )
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)
            continue

        if resp.status_code in (403, 429):
            resp.raise_for_status()
        if resp.status_code >= 500:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            time.sleep(2**attempt)
            continue

        resp.raise_for_status()
        results = resp.json()
        break

    if not results:
        return dict(EMPTY_NOMINATIM_RESULT)

    top = results[0]
    address = top.get("address", {})
    city = next((address[f] for f in CITY_FIELDS if address.get(f)), None)
    state = next((address[f] for f in STATE_FIELDS if address.get(f)), None)

    return {
        "nominatim_found": True,
        "nominatim_city": city,
        "nominatim_state": state,
        "nominatim_country": address.get("country"),
        "nominatim_country_code": address.get("country_code"),
        "nominatim_lat": top.get("lat"),
        "nominatim_lon": top.get("lon"),
        "nominatim_display_name": top.get("display_name"),
    }


def _available_columns(input_csv, wanted):
    """Return the subset of `wanted` columns actually present in
    input_csv's header, for backward-compat reads against older CSVs that
    predate a given column."""
    header = pd.read_csv(input_csv, nrows=0)
    return [c for c in wanted if c in header.columns]


def run(input_csv, output_csv, email, user_agent, sleep_seconds, limit=None):
    print(f"Loading {input_csv}")
    usecols = _available_columns(
        input_csv, ["places", "places_clean", "place_name_008", "place_752"]
    )
    df = pd.read_csv(input_csv, usecols=usecols)
    places = build_places(df)
    print(f"{len(places)} unique (places_clean, place_name_008) groups total.")

    done = already_processed(output_csv)
    remaining = places[~places["geo_key"].astype(str).isin(done)]
    print(f"{len(done)} already done, {len(remaining)} remaining.")

    if limit is not None:
        remaining = remaining.iloc[:limit]
        print(f"Limiting this run to {len(remaining)} groups.")

    is_new_file = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    session = requests.Session()
    n_done = 0
    # places_clean -> Nominatim result, seeded from already-written rows so a
    # resumed run doesn't repeat a query for text already geocoded under a
    # different geo_key (see module docstring / Nominatim usage policy).
    cache = _seed_places_clean_cache(output_csv)
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if is_new_file:
            writer.writeheader()
            f.flush()

        try:
            for i, row in remaining.reset_index(drop=True).iterrows():
                place = row["places_clean"]
                cached = place in cache
                if cached:
                    result = cache[place]
                    print(
                        f"[{i + 1}/{len(remaining)}] Reusing cached result for "
                        f"{place!r} (geo_key={row['geo_key']!r})"
                    )
                else:
                    print(f"[{i + 1}/{len(remaining)}] Geocoding: {place!r}")
                    result = geocode_place(session, place, email, user_agent)
                    cache[place] = result
                writer.writerow(
                    {
                        "geo_key": row["geo_key"],
                        "places_clean": place,
                        "place_name_008": row["place_name_008"],
                        "place_752": row["place_752"],
                        "places_original_example": row["places_original_example"],
                        "n_records": row["n_records"],
                        **result,
                    }
                )
                f.flush()
                n_done += 1
                if not cached and i < len(remaining) - 1:
                    time.sleep(sleep_seconds)
        except requests.HTTPError as e:
            print(
                f"Stopping early after {n_done}/{len(remaining)} requests this run "
                f"due to HTTP error: {e}. Rerun the same command later to resume."
            )
            return
        except KeyboardInterrupt:
            print(
                f"Interrupted after {n_done}/{len(remaining)} requests this run. "
                "Rerun the same command later to resume."
            )
            return

    print(f"Done. Wrote {n_done} new rows to {output_csv}.")


def _places_match(row, llm_result):
    """True iff the places_clean-based and llm_normalized_place-based
    Nominatim lookups both succeeded and resolved to the same
    city/state/country."""
    if not row.get("nominatim_found") or not llm_result["nominatim_found"]:
        return False
    for field in ("nominatim_city", "nominatim_state", "nominatim_country"):
        clean_value = row.get(field)
        clean_value = None if pd.isna(clean_value) else clean_value
        if clean_value != llm_result[field]:
            return False
    return True


def run_llm(input_csv, output_csv, email, user_agent, sleep_seconds, limit=None):
    """Geocode the `llm_normalized_place` column of `imprints.llm_geocode`'s
    output via Nominatim, alongside the `places_clean`-based Nominatim
    columns it already carries. Each geo_key group is geocoded independently
    (unlike `run()`, no places_clean-level caching applies here: distinct
    groups sharing a places_clean are expected to carry distinct
    llm_normalized_place guesses once the 008 hint has taken effect)."""
    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv).sort_values("geo_key").reset_index(drop=True)
    print(f"{len(df)} places total.")

    done = already_processed(output_csv)
    remaining = df[~df["geo_key"].astype(str).isin(done)]
    print(f"{len(done)} already done, {len(remaining)} remaining.")

    if limit is not None:
        remaining = remaining.iloc[:limit]
        print(f"Limiting this run to {len(remaining)} places.")

    is_new_file = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    session = requests.Session()
    n_done = 0
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LLM_OUTPUT_FIELDS)
        if is_new_file:
            writer.writeheader()
            f.flush()

        try:
            for i, row in remaining.reset_index(drop=True).iterrows():
                llm_place = row["llm_normalized_place"]
                has_llm_place = pd.notna(llm_place) and str(llm_place).strip() != ""

                if has_llm_place:
                    print(f"[{i + 1}/{len(remaining)}] Geocoding: {llm_place!r}")
                    result = geocode_place(session, llm_place, email, user_agent)
                else:
                    print(
                        f"[{i + 1}/{len(remaining)}] Skipping (no "
                        f"llm_normalized_place): {row['places_clean']!r}"
                    )
                    result = dict(EMPTY_NOMINATIM_RESULT)

                writer.writerow(
                    {
                        "geo_key": row["geo_key"],
                        "places_clean": row["places_clean"],
                        "place_name_008": row.get("place_name_008"),
                        "place_752": row.get("place_752"),
                        "places_original_example": row["places_original_example"],
                        "n_records": row["n_records"],
                        "nominatim_found": row.get("nominatim_found"),
                        "nominatim_city": row.get("nominatim_city"),
                        "nominatim_state": row.get("nominatim_state"),
                        "nominatim_country": row.get("nominatim_country"),
                        "llm_normalized_place": row.get("llm_normalized_place"),
                        "llm_model": row.get("llm_model"),
                        "llm_nominatim_found": result["nominatim_found"],
                        "llm_nominatim_city": result["nominatim_city"],
                        "llm_nominatim_state": result["nominatim_state"],
                        "llm_nominatim_country": result["nominatim_country"],
                        "llm_nominatim_country_code": result["nominatim_country_code"],
                        "llm_nominatim_lat": result["nominatim_lat"],
                        "llm_nominatim_lon": result["nominatim_lon"],
                        "llm_nominatim_display_name": result["nominatim_display_name"],
                        "nominatim_match": _places_match(row, result),
                    }
                )
                f.flush()
                n_done += 1
                if has_llm_place and i < len(remaining) - 1:
                    time.sleep(sleep_seconds)
        except requests.HTTPError as e:
            print(
                f"Stopping early after {n_done}/{len(remaining)} requests this run "
                f"due to HTTP error: {e}. Rerun the same command later to resume."
            )
            return
        except KeyboardInterrupt:
            print(
                f"Interrupted after {n_done}/{len(remaining)} requests this run. "
                "Rerun the same command later to resume."
            )
            return

    print(f"Done. Wrote {n_done} new rows to {output_csv}.")


def main():
    parser = argparse.ArgumentParser(
        description="Geocode places of publication via Nominatim. Resumable: "
        "rerun the same command to continue after an interruption or API "
        "block."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["places", "llm"],
        default="places",
        help="'places' (default) geocodes each unique (places_clean, "
        "place_name_008) group from --input_csv (data.csv-shaped). 'llm' "
        "geocodes the llm_normalized_place column of imprints.llm_geocode's "
        "output instead, and adds a nominatim_match column comparing it "
        "against the places_clean-based Nominatim result already in that "
        "file.",
    )
    parser.add_argument(
        "--input_csv",
        default=None,
        help="Default: data/PS/data.csv for mode=places, "
        "data/PS/llm_geocode.csv for mode=llm.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Default: data/PS/nominatim_full.csv for mode=places, "
        "data/PS/llm_geocode_nominatim.csv for mode=llm.",
    )
    parser.add_argument(
        "--email",
        required=True,
        help="Contact email sent to Nominatim to identify the requests.",
    )
    parser.add_argument(
        "--user_agent",
        default=None,
        help="Custom User-Agent header (defaults to an imprints-research string "
        "using --email).",
    )
    parser.add_argument(
        "--sleep_seconds",
        type=float,
        default=1.1,
        help="Delay between requests (Nominatim policy: max 1/second).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many new places this run, then stop "
        "(useful for chunked/test runs). Default: process all remaining.",
    )
    args = parser.parse_args()

    user_agent = args.user_agent or f"imprints-research/0.1 (contact: {args.email})"

    if args.mode == "places":
        input_csv = args.input_csv or "data/PS/data.csv"
        output_csv = args.output_csv or "data/PS/nominatim_full.csv"
        run(
            input_csv,
            output_csv,
            args.email,
            user_agent,
            args.sleep_seconds,
            args.limit,
        )
    else:
        input_csv = args.input_csv or "data/PS/llm_geocode.csv"
        output_csv = args.output_csv or "data/PS/llm_geocode_nominatim.csv"
        run_llm(
            input_csv,
            output_csv,
            args.email,
            user_agent,
            args.sleep_seconds,
            args.limit,
        )


if __name__ == "__main__":
    main()
