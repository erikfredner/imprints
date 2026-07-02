"""
Extract sorted unique places of publication for PS records, excluding
PS8000-8999 (Canadian literature in English under LC classification --
outside this project's scope of tracking *US* literary publishing, per
CLAUDE.md).

Reads directly from the raw imprints.data_collection pickles (list-of-dict
records with "classifications" and "places" fields), not from the cleaned
data/PS/data.csv, so it works independently of whether imprints.data_cleaning
has been run. Only basic string normalization is applied to place strings
(lowercasing, whitespace stripping/collapsing) -- deliberately not the
repo's imprints.data_cleaning.clean_string / expand_places pipeline used
elsewhere, so entries retain raw MARC ISBD punctuation (e.g. "new york :").

Usage:
    python -m imprints.get_ps_places --input_dir data/PS --output_txt data/PS/ps_places.txt
"""

import argparse
import os
import pickle
import re

from imprints.data_collection import parse_class

PS8000_CUTOFF = 8000  # PS8000-8999 = Canadian literature in English; outside scope


def is_target_ps(classifications):
    """True if the first PS-prefixed classification has digits < 8000 (or no digits)."""
    for c in classifications or []:
        if not isinstance(c, str):
            continue
        cls, num = parse_class(c.strip())
        if cls == "PS":
            return num is None or num < PS8000_CUTOFF
    return False


def basic_clean(s):
    """Lowercase, strip surrounding whitespace, collapse internal whitespace."""
    if not isinstance(s, str):
        return None
    s = re.sub(r"\s+", " ", s.strip().lower())
    return s or None


def get_ps_places(input_dir, output_txt):
    pkl_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".pkl"))
    if not pkl_files:
        raise RuntimeError(f"No .pkl files found in {input_dir}")

    unique_places = set()
    for file_name in pkl_files:
        with open(os.path.join(input_dir, file_name), "rb") as f:
            records = pickle.load(f)
        for record in records:
            if not is_target_ps(record.get("classifications")):
                continue
            for place in record.get("places") or []:
                cleaned = basic_clean(place)
                if cleaned:
                    unique_places.add(cleaned)

    sorted_places = sorted(unique_places)
    with open(output_txt, "w", encoding="utf-8") as f:
        for place in sorted_places:
            f.write(f"{place}\n")
    return sorted_places


def main():
    parser = argparse.ArgumentParser(
        description="Extract sorted unique places of publication for PS records "
        "(excluding PS8000+), reading directly from imprints.data_collection pickles."
    )
    parser.add_argument("--input_dir", required=True, help="Directory of .pkl files from imprints.data_collection")
    parser.add_argument("--output_txt", required=True, help="Path to write unique places (one per line).")
    args = parser.parse_args()

    unique_places = get_ps_places(args.input_dir, args.output_txt)
    print(f"Wrote {len(unique_places)} unique places to {args.output_txt}")


if __name__ == "__main__":
    main()
