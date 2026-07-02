"""
Normalize places of publication via an LLM, as a second pass alongside
`imprints.geocode_sample`'s Nominatim lookups.

Nominatim is a string-matching gazetteer: it has no world knowledge, so it
can't disambiguate "Santa Fe" (almost certainly Santa Fe, New Mexico, given
these are US literary-publishing records) from Santa Fe, Argentina, and it
has no way to flag "this looks like a cataloging error, not a real place."
This script asks an LLM to make that judgment call instead, using the
PS/American-literature context plus both the cleaned and original place
strings as evidence, and to return a `null` guess rather than a fabricated
one when the evidence doesn't clearly support a real place of publication.

Takes the same per-unique-place rows `geocode_sample.py` produces (grouped by
`places_clean`, with a representative `places_original_example` and
`n_records` count) and carries the `nominatim_*` columns through to the
output so LLM and Nominatim guesses can be reviewed side by side.

Resumable like `geocode_sample.py`: results are appended to --output_csv one
row at a time as they're produced, and on startup any places_clean values
already present in an existing --output_csv are skipped. If --sample_size is
given, a deterministic subset is drawn first (via --seed) and then filtered
by already-processed rows, so re-running after an interruption resumes
within the same sample rather than drawing a new one.

Unlike geocode_sample.py, OpenAI's rate limits are generous enough (this was
built against 10,000 RPM / 10M TPM for gpt-5.4-mini-2026-03-17) that requests
are fired concurrently via a thread pool (--concurrency, default 50) rather
than one at a time. A row that fails after its retries is skipped (not
written) rather than stopping the whole run, so it's naturally picked up on
the next resume; only a KeyboardInterrupt stops the run early.

Usage (full dataset, default --input_csv is the full nominatim output):
    python -m imprints.llm_geocode

Usage (100-row test sample):
    python -m imprints.llm_geocode --sample_size 100 --seed 42
"""

import argparse
import csv
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import APIError, OpenAI
from pydantic import BaseModel, Field

INSTRUCTIONS = """\
You are normalizing places of publication extracted from Library of \
Congress MARC catalog records in the PS classification range (American/US \
literature). Each input is a place string as recorded by a cataloger: it \
may be abbreviated, use inconsistent punctuation, or contain a cataloging \
error or typo.

Identify the single most plausible real-world place of publication and \
return it lowercase, with no accented characters, as:
- "city, state" for US places, using the full state name and never an \
abbreviation (e.g. "baton rouge, louisiana", not "baton rouge, la")
- "city, country" or "city, state, country" for non-US places (country is \
optional only when the place is in a US state)

Because these are records of American literary publishing, prefer a US \
reading when a city name is ambiguous between a US place and a foreign one \
of the same name, unless there is a well-known reason otherwise -- e.g. \
"Santa Fe" is almost certainly Santa Fe, New Mexico rather than Santa Fe, \
Argentina, but "London" most likely refers to London, UK and "Paris" to \
Paris, France, since those are not common American place names. This does \
not mean every record is from the US: foreign places of publication are \
common and expected.

If the input is too garbled, generic ("various places", "s l", "n p"), or \
does not clearly correspond to a real, identifiable place, return null \
rather than guessing.
"""

OUTPUT_FIELDS = [
    "places_clean",
    "places_original_example",
    "n_records",
    "nominatim_found",
    "nominatim_city",
    "nominatim_state",
    "nominatim_country",
    "llm_normalized_place",
    "llm_model",
]


class PlaceNormalization(BaseModel):
    normalized_place: str | None = Field(
        description=(
            "Best-guess real place of publication, lowercase, no accented "
            "characters. US places as 'city, state' with the full state "
            "name (never an abbreviation, e.g. 'baton rouge, louisiana' not "
            "'baton rouge, la'). Non-US places as 'city, country' or 'city, "
            "state, country'. Null if the input does not clearly "
            "correspond to a real, plausible place of publication."
        )
    )


def already_processed(output_csv):
    """Return the set of places_clean values already recorded in output_csv."""
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set()
    done = pd.read_csv(output_csv, usecols=["places_clean"])
    return set(done["places_clean"].astype(str))


def normalize_place(
    client, model, places_clean, places_original_example, max_retries=3
):
    """Ask the model for a normalized place guess for one places_clean value.

    Returns the parsed normalized_place string (or None). Raises
    openai.APIError after max_retries so the caller can stop the run rather
    than continue against a persistent failure.
    """
    user_input = (
        f"places_clean: {places_clean}\n"
        f"places_original_example: {places_original_example!r}"
    )

    for attempt in range(max_retries):
        try:
            response = client.responses.parse(
                model=model,
                instructions=INSTRUCTIONS,
                input=user_input,
                text_format=PlaceNormalization,
            )
            return response.output_parsed.normalized_place
        except APIError:
            if attempt == max_retries - 1:
                raise
            time.sleep(2**attempt)

    return None


def _row_to_output(row, model, llm_normalized_place):
    return {
        "places_clean": row["places_clean"],
        "places_original_example": row["places_original_example"],
        "n_records": row["n_records"],
        "nominatim_found": row.get("nominatim_found"),
        "nominatim_city": row.get("nominatim_city"),
        "nominatim_state": row.get("nominatim_state"),
        "nominatim_country": row.get("nominatim_country"),
        "llm_normalized_place": llm_normalized_place,
        "llm_model": model,
    }


def run(input_csv, output_csv, model, sample_size, seed, limit, concurrency):
    load_dotenv()
    client = OpenAI()

    print(f"Loading {input_csv}")
    df = pd.read_csv(input_csv)

    if sample_size is not None:
        df = df.sample(n=sample_size, random_state=seed).sort_values("places_clean")
        print(f"Drew a random sample of {len(df)} places (seed={seed}).")

    done = already_processed(output_csv)
    remaining = df[~df["places_clean"].astype(str).isin(done)]
    print(f"{len(done)} already done, {len(remaining)} remaining.")

    if limit is not None:
        remaining = remaining.iloc[:limit]
        print(f"Limiting this run to {len(remaining)} places.")

    if remaining.empty:
        print("Nothing to do.")
        return

    total = len(remaining)
    is_new_file = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
    write_lock = threading.Lock()
    n_done = 0
    n_failed = 0

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        if is_new_file:
            writer.writeheader()
            f.flush()

        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures = {
            executor.submit(
                normalize_place,
                client,
                model,
                row["places_clean"],
                row["places_original_example"],
            ): row
            for _, row in remaining.iterrows()
        }

        try:
            for future in as_completed(futures):
                row = futures[future]
                place = row["places_clean"]
                try:
                    llm_normalized_place = future.result()
                except Exception as e:
                    n_failed += 1
                    print(
                        f"[{n_done + n_failed}/{total}] FAILED {place!r}: {e} "
                        "(will retry on next run)"
                    )
                    continue

                with write_lock:
                    writer.writerow(_row_to_output(row, model, llm_normalized_place))
                    f.flush()
                n_done += 1
                print(
                    f"[{n_done + n_failed}/{total}] {place!r} -> {llm_normalized_place!r}"
                )
        except KeyboardInterrupt:
            print(
                f"Interrupted after {n_done}/{total} ({n_failed} failed) this run. "
                "Rerun the same command later to resume."
            )
            executor.shutdown(wait=False, cancel_futures=True)
            return

        executor.shutdown(wait=True)

    print(
        f"Done. Wrote {n_done} new rows to {output_csv} "
        f"({n_failed} failed and will be retried on the next run)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Normalize places of publication via an LLM. Resumable: "
        "rerun the same command to continue after an interruption or error."
    )
    parser.add_argument("--input_csv", default="data/PS/nominatim_full.csv")
    parser.add_argument("--output_csv", default="data/PS/llm_geocode.csv")
    parser.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Draw a random sample of this many places instead of "
        "processing the whole input. Default: process all remaining.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --sample_size (for a reproducible sample).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many new places this run, then stop "
        "(useful for chunked/test runs). Default: process all remaining.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Number of concurrent API requests (thread pool size). "
        "Default 50 stays well under typical rate limits (e.g. 10,000 RPM); "
        "raise it if your account limits allow.",
    )
    args = parser.parse_args()

    run(
        args.input_csv,
        args.output_csv,
        args.model,
        args.sample_size,
        args.seed,
        args.limit,
        args.concurrency,
    )


if __name__ == "__main__":
    main()
