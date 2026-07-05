"""
Build a disambiguating geocoding key and LLM hint text from `places_clean`
plus the MARC 008 place-of-publication signal (`imprints.marc_places`).

`places_clean` alone groups records purely by their (mechanically cleaned)
place-of-publication string, which conflates distinct real places that share
a bare city name -- "Athens" (Georgia, Ohio, Illinois, or Greece) or
"Columbia" (Missouri, South Carolina, ...) all clean to the same string.
`place_name_008`, decoded from MARC 008 bytes 15-17, is present on ~94% of
PS-range records and varies within these groups, so combining it with
`places_clean` splits an ambiguous group into per-place-of-publication
subgroups wherever the signal is available. Kept as a single shared module
so every producer/consumer of this key (geocode_sample, llm_geocode,
join_geocoded) builds it identically -- the class-range-parsing duplication
elsewhere in this codebase is exactly the kind of drift this avoids.

`country_codes_044`/`country_names_044` are not part of this key: prevalence
analysis of data/PS/data.csv found 044 present on only 0.005% of records
(vs. 94.3% for 008), too rare to justify building grouping/fan-out around.
`place_752` (0.067% prevalence) is similarly too rare to key on, but when
present it's already human-readable hierarchical text, so it's surfaced as
extra LLM context via `build_place_hint` rather than folded into the key.
"""

import pandas as pd


def _normalize(value):
    """Coerce a possibly-missing scalar (None, NaN, empty string) to ''."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def build_geo_key(places_clean, place_name_008):
    """Return the composite grouping/join key for a place-of-publication row.

    Groups sharing the same `places_clean` but a different (decoded, non-
    empty) `place_name_008` get distinct keys; rows with no 008 signal key
    identically to `places_clean` alone, so behavior for the ~5.7% of
    records without it is unchanged from the pre-existing places_clean-only
    grouping.
    """
    places_clean = _normalize(places_clean)
    place_name_008 = _normalize(place_name_008)
    return f"{places_clean}||{place_name_008}"


def build_place_hint(place_name_008, place_752_example=None):
    """Return human-readable MARC-signal hint text for the LLM prompt, or
    None if neither signal is available.

    `place_name_008` is treated as the primary hint (94% prevalence,
    demonstrated to discriminate ambiguous city names). `place_752_example`,
    when given, is a representative flattened 752 hierarchy string (see
    `imprints.data_cleaning._flatten_place_hierarchy`) added as a second,
    corroborating line -- it is rare (well under 1% of records) but already
    full place text when present.
    """
    place_name_008 = _normalize(place_name_008)
    place_752_example = _normalize(place_752_example)

    lines = []
    if place_name_008:
        lines.append(f"marc_008_place_hint: {place_name_008}")
    if place_752_example:
        lines.append(f"marc_752_place_hint: {place_752_example}")
    return "\n".join(lines) if lines else None
