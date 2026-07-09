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

For a multi-place record whose existing curated `city_group` label is
``"New York City"``, a single record-level 008 scope can describe another
component of the co-publication (for example, ``London | New York`` with an
England 008 code). Those NYC components receive a separate key policy so
they cannot be conflated with ordinary ``new york||England`` rows. The
direct geocoder still tries 008 first, then can fall back to the curated NYC
label only when that scoped result conflicts.

`country_codes_044`/`country_names_044` are not part of this key: prevalence
analysis of data/PS/data.csv found 044 present on only 0.005% of records
(vs. 94.3% for 008), too rare to justify building grouping/fan-out around.
`place_752` (0.067% prevalence) is similarly too rare to key on, but when
present it's already human-readable hierarchical text, so it's surfaced as
extra LLM context via `build_place_hint` rather than folded into the key.
"""

import ast

import pandas as pd


NYC_GROUP = "New York City"
NO_PLACE_GROUP = "No place of publication"

# The ordinary key remains byte-for-byte compatible with prior outputs. The
# candidate suffix exists only to keep multi-place NYC rows separate from
# otherwise identical 008-scoped rows during direct geocoding and joining.
GEO_KEY_POLICY_008 = "008"
GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE = "nyc_multiplace_candidate"


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


def build_geo_key(
    places_clean,
    place_name_008,
    key_policy: str = GEO_KEY_POLICY_008,
):
    """Return the composite grouping/join key for a place-of-publication row.

    Groups sharing the same `places_clean` but a different (decoded, non-
    empty) `place_name_008` get distinct keys; rows with no 008 signal key
    identically to `places_clean` alone, so behavior for the ~5.7% of
    records without it is unchanged from the pre-existing places_clean-only
    grouping. ``key_policy`` is normally ``"008"`` and preserves that
    historical key. A multi-place NYC candidate receives a suffix so it is
    not grouped with a non-candidate row carrying the same 008 signal.
    """
    places_clean = _normalize(places_clean)
    place_name_008 = _normalize(place_name_008)
    base = f"{places_clean}||{place_name_008}"
    if key_policy == GEO_KEY_POLICY_008:
        return base
    return f"{base}||{key_policy}"


def add_geocode_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``geo_key`` and ``geo_key_policy`` columns.

    The normal key is based only on ``places_clean`` and ``place_name_008``.
    If the data contains the record identifier and curated city label, an NYC
    component is marked as a candidate when its LCCN has more than one
    distinct, real publication place. The marker changes the key, but does
    *not* itself override the 008 signal; that decision belongs to
    :func:`imprints.geonames_geocode.run_direct`, after it has attempted the
    regular scoped match.

    Inputs lacking ``lccn`` or ``city_group`` retain the ordinary key. This
    keeps older CSV-shaped inputs readable while making the policy explicit
    for current pipeline outputs.
    """
    out = df.copy()
    if "place_name_008" not in out.columns:
        out["place_name_008"] = None

    out["geo_key_policy"] = GEO_KEY_POLICY_008
    required = {"lccn", "places_clean", "city_group"}
    if required.issubset(out.columns):
        real_place = (
            out["lccn"].notna()
            & out["places_clean"].notna()
            & out["places_clean"].astype(str).str.strip().ne("")
            & out["city_group"].ne(NO_PLACE_GROUP)
        )
        place_counts = (
            out.loc[real_place]
            .groupby("lccn")["places_clean"]
            .nunique()
        )
        record_place_count = out["lccn"].map(place_counts).fillna(0)
        candidate = (
            out["city_group"].eq(NYC_GROUP) & record_place_count.gt(1)
        )
        out.loc[candidate, "geo_key_policy"] = (
            GEO_KEY_POLICY_NYC_MULTIPLACE_CANDIDATE
        )

    out["geo_key"] = [
        build_geo_key(places_clean, place_name_008, key_policy)
        for places_clean, place_name_008, key_policy in zip(
            out["places_clean"], out["place_name_008"], out["geo_key_policy"]
        )
    ]
    return out


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
    """Group by geo_key, keeping one representative value and record count.

    The ordinary key combines ``places_clean`` and ``place_name_008``; a
    multi-place NYC candidate adds its policy suffix. A missing
    ``place_name_008`` column (older data.csv, pre-008 capture) is treated as
    all-missing, so non-candidate rows fall back to places_clean alone.
    """
    df = df[df["places"].notna() & df["places_clean"].notna()]
    df = df[df["places_clean"].astype(str).str.strip() != ""]
    if "place_752" not in df.columns:
        df = df.assign(place_752=None)

    df = add_geocode_key_columns(df)

    grouped = (
        df.groupby("geo_key")
        .agg(
            places_clean=("places_clean", "first"),
            place_name_008=("place_name_008", "first"),
            geo_key_policy=("geo_key_policy", "first"),
            place_752=("place_752", _first_non_null),
            places_original_example=("places", "first"),
            n_records=("places", "size"),
        )
        .reset_index()
    )
    grouped["place_752"] = grouped["place_752"].map(_first_place_752_example)
    return grouped.sort_values("geo_key").reset_index(drop=True)
