"""
Canonicalize place-of-publication strings for "unique places" analyses.

`clean_string` (imprints.data_cleaning) only lowercases and strips
punctuation -- it doesn't know that "boston ma", "boston mass", and "boston
massachusetts" name the same city, or that "s l" / "n p" are MARC
abbreviations for "no place of publication," not places. That makes
``places_clean`` unsuitable, as-is, for counting *distinct* publication
locations: the same city fragments across cataloging-convention variants,
and non-place markers get counted as places.

This module collapses US state-name variants (postal code, spelled out, or
traditional abbreviation -- e.g. "mass"/"massachusetts"/"ma") to a single
canonical "<city> <postal code>" form, drops known no-place markers, and
truncates address noise that sometimes follows the state token in
self-published/vanity-press records (e.g. "abilene tex ligustrum dr
abilene" -> "abilene tex" -> "abilene tx").

It intentionally does NOT merge a bare city name (e.g. "boston") with its
state-qualified form ("boston ma") -- doing so would require assuming no
other city shares that name, which isn't always safe. It also does not
attempt non-US place normalization, or address noise lacking a US state
marker (e.g. "san francisco divisadero san francisco"). Both are smaller,
harder-to-resolve residuals left as understood limitations.
"""

from imprints.data_cleaning import clean_string

# Unclean variant spellings for each USPS state/territory code. Cleaned with
# clean_string below so tokens match places_clean exactly (punctuation
# stripped, lowercased). The postal code itself is added automatically.
_RAW_STATE_VARIANTS = {
    "AL": ["Alabama", "Ala"],
    "AK": ["Alaska"],
    "AZ": ["Arizona", "Ariz"],
    "AR": ["Arkansas", "Ark"],
    "CA": ["California", "Calif", "Cal"],
    "CO": ["Colorado", "Colo"],
    "CT": ["Connecticut", "Conn"],
    "DE": ["Delaware", "Del"],
    "DC": ["District of Columbia", "D.C."],
    "FL": ["Florida", "Fla"],
    "GA": ["Georgia", "Ga"],
    "HI": ["Hawaii"],
    "ID": ["Idaho"],
    "IL": ["Illinois", "Ill"],
    "IN": ["Indiana", "Ind"],
    "IA": ["Iowa"],
    "KS": ["Kansas", "Kan", "Kans"],
    "KY": ["Kentucky", "Ky"],
    "LA": ["Louisiana", "La"],
    "ME": ["Maine"],
    "MD": ["Maryland", "Md"],
    "MA": ["Massachusetts", "Mass"],
    "MI": ["Michigan", "Mich"],
    "MN": ["Minnesota", "Minn"],
    "MS": ["Mississippi", "Miss"],
    "MO": ["Missouri", "Mo"],
    "MT": ["Montana", "Mont"],
    "NE": ["Nebraska", "Neb", "Nebr"],
    "NV": ["Nevada", "Nev"],
    "NH": ["New Hampshire", "N.H."],
    "NJ": ["New Jersey", "N.J."],
    "NM": ["New Mexico", "N.M.", "N. Mex."],
    "NY": ["New York", "N.Y."],
    "NC": ["North Carolina", "N.C."],
    "ND": ["North Dakota", "N.D.", "N. Dak."],
    "OH": ["Ohio"],
    "OK": ["Oklahoma", "Okla"],
    "OR": ["Oregon", "Ore", "Oreg"],
    "PA": ["Pennsylvania", "Pa", "Penn", "Penna"],
    "PR": ["Puerto Rico", "P.R."],
    "RI": ["Rhode Island", "R.I."],
    "SC": ["South Carolina", "S.C."],
    "SD": ["South Dakota", "S.D.", "S. Dak."],
    "TN": ["Tennessee", "Tenn"],
    "TX": ["Texas", "Tex"],
    "UT": ["Utah"],
    "VT": ["Vermont", "Vt"],
    "VA": ["Virginia", "Va"],
    # Bare "Washington" deliberately excluded: it's a common place-name
    # prefix in its own right (Washington Square, Washington Court House,
    # OH, Washington DC) far more often than it's the spelled-out state
    # name in this corpus, so matching it would misclassify real places as
    # bare WA state.
    "WA": ["Wash"],
    "WV": ["West Virginia", "W.Va.", "W. Va."],
    "WI": ["Wisconsin", "Wis", "Wisc"],
    "WY": ["Wyoming", "Wyo"],
}

# MARC/cataloger conventions for "we don't know the place" -- not places.
NO_PLACE_MARKERS = {
    clean_string(s)
    for s in [
        "s.l.",
        "n.p.",
        "s.n.",
        "place of publication not identified",
        "publisher location not identified",
        "unknown place of publication",
    ]
}


def _build_state_token_map():
    """Map each cleaned variant token-tuple to its canonical postal code."""
    token_map = {}
    for code, variants in _RAW_STATE_VARIANTS.items():
        for variant in [*variants, code]:
            cleaned = clean_string(variant)
            if cleaned:
                token_map[tuple(cleaned.split())] = code.lower()
    return token_map


_STATE_TOKEN_MAP = _build_state_token_map()
_MAX_STATE_TOKENS = max(len(key) for key in _STATE_TOKEN_MAP)


def _match_state_at(tokens, start):
    """Return (postal_code, end_index) for the longest state match starting
    at tokens[start], or None."""
    for length in range(_MAX_STATE_TOKENS, 0, -1):
        candidate = tuple(tokens[start : start + length])
        if candidate in _STATE_TOKEN_MAP:
            return _STATE_TOKEN_MAP[candidate], start + length
    return None


def canonicalize_place(places_clean):
    """Collapse a cleaned place string to a canonical city/state form.

    Returns None for known no-place markers. Otherwise finds the LAST
    recognized US state token in the string, drops anything after it
    (address noise), and returns "<city tokens> <postal code>". Preferring
    the last match (rather than the first) correctly handles the case where
    a city is itself named after a state, e.g. "delaware ohio" -> "delaware
    oh" rather than misreading "delaware" as the state and pointer stopping
    there. A state-only string (no city tokens before the match) returns the
    bare postal code. Strings with no recognized state token -- bare city
    names, non-US places, and address noise lacking a state marker -- pass
    through unchanged.
    """
    if places_clean is None:
        return None
    text = str(places_clean).strip()
    if not text:
        return None
    if text in NO_PLACE_MARKERS:
        return None

    tokens = text.split()
    last_match = None
    for start in range(len(tokens)):
        match = _match_state_at(tokens, start)
        if match is None:
            continue
        # "mt" leading a multi-token string is virtually always "Mount ___"
        # (Mt. Horeb, Mt. Vernon...), not Montana's postal code -- a real
        # "City, MT" record always has the city before "mt", never at
        # position 0. Skip this specific collision rather than mislabel the
        # place as bare Montana.
        if start == 0 and len(tokens) > 1 and tokens[0] == "mt":
            continue
        last_match = (start, match[0])
    if last_match is None:
        return text

    start, code = last_match
    city_tokens = tokens[:start]
    return f"{' '.join(city_tokens)} {code}".strip()
