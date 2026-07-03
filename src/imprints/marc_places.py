"""
Decode MARC country/place codes (used by 008 bytes 15-17 and 044 $a) into
human-readable names.

Both fields share the same MARC country code system (the "MARC List of
Countries"), so one reference table serves both. `marc_country_codes.csv`
was generated from the Library of Congress's official codelist
(https://www.loc.gov/standards/codelists/countries.xml) and includes
obsolete/historical codes (e.g. "xxr" -> Soviet Union, "cs" -> Czechoslovakia,
"yu" -> Yugoslavia) alongside current ones, since the source MARC records
span 1945-2019 and older records may carry a code that was current at their
own cataloging date rather than today's. Where an obsolete code collides
with a code that is *currently* active for a different place (e.g. "ai" is
both Armenia's live code and Anguilla's discontinued pre-1988 code), the
currently active meaning wins.

044 $c uses a different code system (ISO 3166) and is intentionally not
decoded here.
"""

import csv
import os
from functools import lru_cache

MARC_COUNTRY_CODES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "marc_country_codes.csv")
)


@lru_cache(maxsize=1)
def _load_marc_country_codes():
    """Load code -> name from marc_country_codes.csv once. Returns {} if the
    file is missing or empty."""
    if not os.path.exists(MARC_COUNTRY_CODES_PATH):
        return {}
    with open(MARC_COUNTRY_CODES_PATH, newline="") as f:
        return {row["code"]: row["name"] for row in csv.DictReader(f)}


def decode_marc_country(code):
    """Return the country/place name for a MARC country code, or None if
    `code` is missing/blank/unrecognized. Never raises."""
    if not code:
        return None
    return _load_marc_country_codes().get(code)
