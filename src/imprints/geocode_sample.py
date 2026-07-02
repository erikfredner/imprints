"""
Geocode every distinct cleaned place of publication via Nominatim.

Spot-checks how well `places_clean` (the string produced by
`imprints.data_cleaning.clean_string`) maps back to real-world places by
resolving each unique value against the Nominatim/OpenStreetMap search API.

Respects the Nominatim usage policy: sequential requests only, an absolute
max of 1 request/second, a custom User-Agent, and no repeated identical
queries (each unique place string is queried exactly once).
See: https://operations.osmfoundation.org/policies/nominatim/

This is a long-running job (order-12k unique places at ~1 req/sec is several
hours), so it's built to resume: results are appended to --output_csv one row
at a time as they're produced, and on startup any places_clean values already
present in an existing --output_csv are skipped. If the API starts rate
limiting/blocking (403/429) or the process is interrupted, just rerun the
same command later -- it picks up where it left off.

Usage:
    python -m imprints.geocode_sample \
        --input_csv data/PS/data.csv \
        --output_csv data/PS/nominatim_full.csv \
        --email you@example.com
"""

import argparse
import csv
import os
import time

import pandas as pd
import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
CITY_FIELDS = ("city", "town", "village", "borough", "suburb")
STATE_FIELDS = ("state", "state_district")
OUTPUT_FIELDS = [
    "places_clean",
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


def build_places(df):
    """Group by places_clean, keeping one representative original value and
    a count per group, sorted for a deterministic, resumable order."""
    df = df[df["places"].notna() & df["places_clean"].notna()]
    df = df[df["places_clean"].astype(str).str.strip() != ""]

    grouped = df.groupby("places_clean").agg(
        places_original_example=("places", "first"),
        n_records=("places", "size"),
    )
    return grouped.reset_index().sort_values("places_clean").reset_index(drop=True)


def already_processed(output_csv):
    """Return the set of places_clean values already recorded in output_csv."""
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set()
    done = pd.read_csv(output_csv, usecols=["places_clean"])
    return set(done["places_clean"].astype(str))


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
        return {
            "nominatim_found": False,
            "nominatim_city": None,
            "nominatim_state": None,
            "nominatim_country": None,
            "nominatim_country_code": None,
            "nominatim_lat": None,
            "nominatim_lon": None,
            "nominatim_display_name": None,
        }

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


def run(input_csv, output_csv, email, user_agent, sleep_seconds, limit=None):
    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv, usecols=["places", "places_clean"])
    places = build_places(df)
    print(f"{len(places)} unique places_clean values total.")

    done = already_processed(output_csv)
    remaining = places[~places["places_clean"].astype(str).isin(done)]
    print(f"{len(done)} already done, {len(remaining)} remaining.")

    if limit is not None:
        remaining = remaining.iloc[:limit]
        print(f"Limiting this run to {len(remaining)} places.")

    is_new_file = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    session = requests.Session()
    n_done = 0
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if is_new_file:
            writer.writeheader()
            f.flush()

        try:
            for i, row in remaining.reset_index(drop=True).iterrows():
                place = row["places_clean"]
                print(f"[{i + 1}/{len(remaining)}] Geocoding: {place!r}")
                result = geocode_place(session, place, email, user_agent)
                writer.writerow(
                    {
                        "places_clean": place,
                        "places_original_example": row["places_original_example"],
                        "n_records": row["n_records"],
                        **result,
                    }
                )
                f.flush()
                n_done += 1
                if i < len(remaining) - 1:
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
        description="Geocode every distinct cleaned place via Nominatim. "
        "Resumable: rerun the same command to continue after an interruption "
        "or API block."
    )
    parser.add_argument("--input_csv", default="data/PS/data.csv")
    parser.add_argument("--output_csv", default="data/PS/nominatim_full.csv")
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

    run(
        args.input_csv,
        args.output_csv,
        args.email,
        user_agent,
        args.sleep_seconds,
        args.limit,
    )


if __name__ == "__main__":
    main()
