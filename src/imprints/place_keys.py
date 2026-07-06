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
so every producer/consumer of this key (geonames_geocode, llm_geocode,
join_geocoded) builds it identically -- the class-range-parsing duplication
elsewhere in this codebase is exactly the kind of drift this avoids.
`build_places` groups raw per-record rows into one row per geo_key
(representative places_clean/place_name_008/original value plus a record
count), the shape every geocoding pass and `llm_geocode` operate on.

`country_codes_044`/`country_names_044` are not part of this key: prevalence
analysis of data/PS/data.csv found 044 present on only 0.005% of records
(vs. 94.3% for 008), too rare to justify building grouping/fan-out around.
`place_752` (0.067% prevalence) is similarly too rare to key on, but when
present it's already human-readable hierarchical text, so it's surfaced as
extra LLM context via `build_place_hint` rather than folded into the key.
"""

import ast

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
            build_geo_key(pc, p8)
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
