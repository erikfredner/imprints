import os
import pandas as pd
import pickle
import re
import unicodedata
from functools import lru_cache

from imprints.marc_places import decode_marc_country

NYC_VARIANTS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "nyc_variants.txt")
)

# ---------------------- Cleaning Functions ----------------------


def parse_class(class_str):
    """
    Split into prefix and digits.
    E.g. PS3555.123 -> ('PS', 3555)
    """
    m = re.match(r"([A-Z]+)(\d+)?", str(class_str))
    if not m:
        return None, None
    return m.group(1), int(m.group(2)) if m.group(2) else None


# LC classification carves Canadian literature (English) out of the PS
# schedule at PS8001-8599 -- a different national literature that happens to
# share the PS prefix, not American literature. This project's "PS" always
# means US literary publishing (see CLAUDE.md), so these numbers are never a
# match under any class_range, including a bare "PS" or a one-letter "P".
# Duplicated in imprints.data_collection.matches_range; keep both in sync.
PS_CANADIAN_LIT_MIN = 8001
PS_CANADIAN_LIT_MAX = 8599


def matches_range(classification, prefix, num_min=None, num_max=None):
    """
    Test if a classification matches a given prefix and optional num range.
    """
    if not classification or not isinstance(classification, str):
        return False
    cls, num = parse_class(classification.strip())
    # A one-letter range is a top-level LC class and includes its subclasses
    # (e.g. P includes PR and PS).  Longer ranges identify a complete alpha
    # subclass, so PS must not also admit malformed/other classes such as PSA.
    prefix_matches = bool(cls) and (
        cls == prefix or (len(prefix) == 1 and cls.startswith(prefix))
    )
    if not prefix_matches:
        return False
    if (
        cls == "PS"
        and num is not None
        and PS_CANADIAN_LIT_MIN <= num <= PS_CANADIAN_LIT_MAX
    ):
        return False
    if num_min is not None and (num is None or num < num_min):
        return False
    if num_max is not None and (num is None or num > num_max):
        return False
    return True


def parse_range_spec(range_str):
    """
    Parse ranges: 'PS' -> {'prefix':'PS'}, 'PR9000-PR9999' -> {'prefix':'PR', 'min':9000, 'max':9999}
    """
    m = re.match(r"^([A-Z]+)(\d{0,})(-([A-Z]+)?(\d{1,}))?$", range_str)
    if not m:
        raise ValueError(f"Range spec not recognized: {range_str}")
    prefix = m.group(1)
    end_prefix = m.group(4)
    if end_prefix and end_prefix != prefix:
        raise ValueError("Range endpoints must use the same LC prefix")
    minval = int(m.group(2)) if m.group(2) else None
    maxval = int(m.group(5)) if m.group(5) else None
    return {"prefix": prefix, "min": minval, "max": maxval}


def filter_classifications(classifications, class_range):
    """
    Given list of classification numbers, return all that match target.
    """
    prefix = class_range["prefix"]
    minval = class_range.get("min")
    maxval = class_range.get("max")
    if not classifications:
        return []
    if isinstance(classifications, str):
        classifications = [classifications]
    matches = []
    for c in classifications:
        try:
            if pd.isna(c):
                continue
        except Exception:
            pass
        c_str = str(c).strip()
        if not c_str:
            continue
        if matches_range(c_str, prefix, minval, maxval):
            matches.append(c_str)
    return matches


def get_digits_for_class(classification, class_range):
    """
    Extract digits from a classification number that matches the prefix.
    """
    if classification is None:
        return None
    try:
        if pd.isna(classification):
            return None
    except Exception:
        pass
    s = str(classification).strip()
    prefix = class_range["prefix"]
    minval = class_range.get("min")
    maxval = class_range.get("max")
    if not matches_range(s, prefix, minval, maxval):
        return None
    _, num = parse_class(s)
    return num


# 4-digit runs in 260/264 $c or publisher text are not always years: street
# numbers ("1420 Chestnut Street"), zip-code fragments ("New York, 10014" ->
# 1001), and typos all produce them. A run embedded in a longer digit string
# is never a year, and this corpus's plausible publication years fall well
# inside this window.
YEAR_MIN_PLAUSIBLE = 1500
YEAR_MAX_PLAUSIBLE = 2030
_YEAR_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
_CORRECTED_YEAR_RE = re.compile(
    r"\bi\.?\s*e\.?\s*[,\s]*(?P<year>\d{4})(?!\d)", re.IGNORECASE
)
_YEAR_RANGE_RE = re.compile(
    r"(?<!\d)(?P<start>\d{4})\s*[-–—/]\s*(?P<end>\d{4})(?!\d)"
)


def get_year_int(year):
    """Extract the first plausible four-digit year from a string (or None)."""
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return None
    text = str(year)
    # Cataloger corrections such as "1863 [i.e. 1963]" supersede the
    # erroneous year printed on the item.
    corrected = _CORRECTED_YEAR_RE.search(text)
    if corrected:
        value = int(corrected.group("year"))
        if YEAR_MIN_PLAUSIBLE <= value <= YEAR_MAX_PLAUSIBLE:
            return value
    # A short span can be a serial/multivolume publication run, for which the
    # start year is useful. A decades-wide span is an uncertainty interval;
    # reporting its lower boundary as an exact publication year is misleading.
    # Restrict this rule to syntactic ranges: prose can legitimately mention
    # distant dates (for example, publication and copyright-renewal years).
    for range_match in _YEAR_RANGE_RE.finditer(text):
        start = int(range_match.group("start"))
        end = int(range_match.group("end"))
        if (
            YEAR_MIN_PLAUSIBLE <= start <= YEAR_MAX_PLAUSIBLE
            and YEAR_MIN_PLAUSIBLE <= end <= YEAR_MAX_PLAUSIBLE
            and abs(end - start) > 25
        ):
            return None
    for match in _YEAR_RE.finditer(text):
        value = int(match.group())
        if YEAR_MIN_PLAUSIBLE <= value <= YEAR_MAX_PLAUSIBLE:
            return value
    return None


def get_years_ints(years):
    # Robust None/NA/empty handling
    if years is None:
        return None
    if isinstance(years, (list, tuple)):
        if not years:
            return None
        years_int = [get_year_int(y) for y in years if get_year_int(y) is not None]
        return min(years_int) if years_int else None
    if hasattr(years, "size") and hasattr(years, "__getitem__"):
        if years.size == 0:
            return None
        years_int = [get_year_int(y) for y in years if get_year_int(y) is not None]
        return min(years_int) if years_int else None
    try:
        if pd.isna(years):
            return None
    except Exception:
        pass
    return get_year_int(str(years))


def get_publishers_year_ints(publishers):
    if publishers is None:
        return None
    if isinstance(publishers, (list, tuple)):
        if not publishers:
            return None
        years = [get_year_int(p) for p in publishers if get_year_int(p) is not None]
        return min(years) if years else None
    if hasattr(publishers, "size") and hasattr(publishers, "__getitem__"):
        if publishers.size == 0:
            return None
        years = [get_year_int(p) for p in publishers if get_year_int(p) is not None]
        return min(years) if years else None
    try:
        if pd.isna(publishers):
            return None
    except Exception:
        pass
    return get_year_int(str(publishers))


def get_decade(year):
    """Convert a year into its decade, e.g. 1983 -> 1980."""
    try:
        year = int(year)
        return (year // 10) * 10 if year > 0 else None
    except (TypeError, ValueError):
        return None


def _parse_place_code_008(field_008):
    """Return the 3-character MARC place-of-publication code at 0-indexed
    bytes 15-17, or None if field_008 is missing/short. Never raises.
    Mirrors imprints.secondary_classification._parse_008's guard convention
    (kept independent rather than imported, since that module is calibrated
    specifically for PS-range secondary-literature classification)."""
    if field_008 is None or (isinstance(field_008, float) and pd.isna(field_008)):
        return None
    if not field_008 or len(field_008) != 40:
        return None
    return field_008[15:18]


def _parse_date1_008(field_008):
    """Return MARC 008 Date 1 (bytes 7-10) when it is a plausible year."""
    if field_008 is None or (isinstance(field_008, float) and pd.isna(field_008)):
        return None
    if not isinstance(field_008, str) or len(field_008) != 40:
        return None
    date_type = field_008[6]
    date1 = get_year_int(field_008[7:11])
    date2 = get_year_int(field_008[11:15])
    # q encodes a range of questionable years, not an exact year. Using its
    # lower bound manufactured spikes at round boundaries (especially 1900).
    if date_type == "q" and date1 != date2:
        return None
    # Some malformed/incomplete "19--" records are encoded as an extremely
    # broad multiple-date range (1900-1999/9999). It is not defensible to
    # report the lower bound as the publication year.
    if date_type == "m" and (
        date2 is None or (date1 is not None and date2 - date1 > 25)
    ):
        return None
    return date1


def _years_from_imprint_fields(occurrences):
    """Return ranked publication, copyright, and other 260/264 years."""
    ranked = {"publication": [], "copyright": [], "other": []}
    if not isinstance(occurrences, (list, tuple)) or not occurrences:
        return ranked
    for occurrence in occurrences:
        tag = occurrence.get("tag")
        ind2 = occurrence.get("ind2", " ")
        if tag == "260" or (tag == "264" and ind2 in (" ", "1")):
            bucket = "publication"
        elif tag == "264" and ind2 == "4":
            bucket = "copyright"
        else:
            bucket = "other"
        for code, text in occurrence.get("subfields", []):
            if code == "c":
                year = get_year_int(text)
                if year is not None:
                    ranked[bucket].append(year)
    return ranked


def _select_publication_year(row):
    """Choose a year by MARC evidence quality and return (year, source)."""
    ranked = _years_from_imprint_fields(row.get("imprint_fields"))
    candidates = (
        (ranked["publication"], "260/264 publication"),
        ([row.get("year_008")] if row.get("year_008") is not None else [], "008 date1"),
        (ranked["copyright"], "264 copyright"),
        (ranked["other"], "264 other"),
        ([row.get("year_int_legacy")] if row.get("year_int_legacy") is not None else [], "legacy 260/264"),
        ([row.get("publisher_year_int")] if row.get("publisher_year_int") is not None else [], "publisher text"),
    )
    for values, source in candidates:
        valid = [int(value) for value in values if value is not None and not pd.isna(value)]
        if valid:
            return min(valid), source
    return None, None


def _decode_country_codes(codes):
    """Order-preserving decode of a list of MARC country codes to names.
    Unrecognized codes decode to None (not dropped), so gaps stay visible
    for QA rather than being silently absorbed. None input -> None."""
    if codes is None:
        return None
    return [decode_marc_country(c) for c in codes]


_PLACE_752_CODES_ORDER = ("a", "b", "c", "d", "e")


def _flatten_place_hierarchy(occurrences):
    """Turn a list of 752 occurrence dicts (imprints.data_collection.
    collect_place_occurrences shape) into a list of human-readable strings,
    one per occurrence, joining $a/$b/$c/$d/$e (country > state/province >
    county > city > city subsection) in that fixed hierarchical order
    regardless of the order subfields appeared in the record. Occurrences
    with no usable subfields are skipped. None/empty input -> None."""
    if not occurrences:
        return None
    flattened = []
    for occ in occurrences:
        by_code = {code: [] for code in _PLACE_752_CODES_ORDER}
        for code, text in occ.get("subfields", []):
            if code in by_code:
                by_code[code].append(text)
        parts = [text for code in _PLACE_752_CODES_ORDER for text in by_code[code]]
        if parts:
            flattened.append(", ".join(parts))
    return flattened or None


def clean_string(s):
    """Lowercase, replace punctuation with spaces, collapse to single spaces."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = "".join(c if c.isalpha() else " " for c in s)
    s = re.sub(r"\s+", " ", s)
    return s.lower().strip()


@lru_cache(maxsize=1)
def _load_nyc_variants():
    """Load and clean NYC variant place names once."""
    with open(NYC_VARIANTS_PATH, "r") as f:
        return {v for v in (clean_string(line) for line in f) if v}


def get_target_cities(placename):
    """Return 'New York City', 'Other', or 'No place of publication' based on NYC variants."""
    if (
        placename is None
        or (isinstance(placename, float) and pd.isna(placename))
        or not str(placename).strip()
    ):
        return "No place of publication"
    placename_clean = clean_string(placename)
    if not placename_clean:
        return "Other"
    return "New York City" if placename_clean in _load_nyc_variants() else "Other"


def flatten_first(item):
    """Get first item if list, else return as is. None if empty but list."""
    if isinstance(item, list):
        return item[0] if item else None
    return item


def normalize_places(value):
    """Ensure places value is list-like and keep rows with missing places."""
    if value is None:
        return [None]
    try:
        if pd.isna(value):
            return [None]
    except Exception:
        pass
    if isinstance(value, (list, tuple)):
        return value if value else [None]
    return [value]


# Separators at the top level join distinct cities in a 260/264 $a, e.g.
# "Boston and New York" or "New York; London". Parentheses/braces group an
# address, but MARC square brackets merely mark cataloger-supplied text and do
# not suppress place separators.
_PLACE_SPLIT_RE = re.compile(r"\s*;\s*|\s*&\s*|\s+and\s+", re.IGNORECASE)
# Records with multiple 260/264 $a occurrences sometimes already arrive as
# separate list items where the continuation entry starts with "and"/"&",
# e.g. ["Boston", "and New York,"] -- a cataloging-side list continuation,
# not a string split_places's own delimiter matching produces (there's no
# leading whitespace left for _PLACE_SPLIT_RE to match once this item is
# handled on its own). Strip it so "and New York" resolves as "New York"
# instead of failing every nyc_variants.txt lookup.
_LEADING_CONNECTOR_RE = re.compile(r"^(?:and|&)\s+", re.IGNORECASE)
_COMPOUND_GEOGRAPHIC_NAME_RE = re.compile(
    r"\b(?:"
    r"antigua\s+(?:and|&)\s+barbuda|"
    r"bosnia\s+(?:and|&)\s+herzegovina|"
    r"sao\s+tome\s+(?:and|&)\s+principe|"
    r"st\.?\s+kitts\s+(?:and|&)\s+nevis|"
    r"saint\s+kitts\s+(?:and|&)\s+nevis|"
    r"st\.?\s+vincent\s+(?:and|&)\s+(?:the\s+)?grenadines|"
    r"saint\s+vincent\s+(?:and|&)\s+(?:the\s+)?grenadines|"
    r"trinidad\s+(?:and|&)\s+tobago|"
    r"turks\s+(?:and|&)\s+caicos|"
    r"wallis\s+(?:and|&)\s+futuna"
    r")\b",
    re.IGNORECASE,
)


def split_places(value):
    """Split top-level city-list separators while preserving nested text.

    Co-publications list multiple cities in one subfield; splitting them lets
    an NYC component be recognized instead of being lost inside a compound
    string like "Boston and New York". Conjunctions inside parentheses or
    braces remain part of one place. None/blank passes through as [None].
    """
    if value is None:
        return [None]
    try:
        if pd.isna(value):
            return [None]
    except Exception:
        pass
    original_text = str(value).strip()
    original_text = _LEADING_CONNECTOR_RE.sub("", original_text).strip()
    if not original_text:
        return [None]
    text = original_text
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    parts = []
    start = cursor = 0
    nesting = 0
    did_split = False
    compound_name_spans = [
        match.span() for match in _COMPOUND_GEOGRAPHIC_NAME_RE.finditer(text)
    ]
    for match in _PLACE_SPLIT_RE.finditer(text):
        for char in text[cursor : match.start()]:
            if char in "({":
                nesting += 1
            elif char in ")}":
                nesting = max(0, nesting - 1)
        inside_compound_name = any(
            compound_start <= match.start() and match.end() <= compound_end
            for compound_start, compound_end in compound_name_spans
        )
        if nesting == 0 and not inside_compound_name:
            part = text[start : match.start()].strip()
            if part:
                parts.append(part)
            start = match.end()
            did_split = True
        cursor = match.end()
    if not did_split:
        return [original_text] if original_text else [None]
    final_part = text[start:].strip()
    if final_part:
        parts.append(final_part)
    return parts or [None]


def expand_places(value):
    """Normalize a record's places value then split each into component cities.

    Produces the list that `cleaning_pipeline` explodes into one row per city,
    order-preserving and with exact raw repeats removed. The pipeline also
    deduplicates after mechanical normalization, so variants such as
    ``New York :`` and ``[New York] :`` collapse too. Missing places are
    preserved as [None].
    """
    expanded = []
    for place in normalize_places(value):
        expanded.extend(split_places(place))
    return list(dict.fromkeys(expanded)) or [None]


def load_pickles_to_dataframe(pickle_dir):
    """Load ALL pickles from directory & concat to single DataFrame."""
    all_data = []
    for file_name in os.listdir(pickle_dir):
        if file_name.endswith(".pkl"):
            file_path = os.path.join(pickle_dir, file_name)
            with open(file_path, "rb") as f:
                data = (
                    pd.read_pickle(f)
                    if file_name.endswith(".df.pkl")
                    else pd.DataFrame(pickle.load(f))
                )
                all_data.append(data)
    if not all_data:
        raise RuntimeError("No pickles found in directory.")
    df = pd.concat(all_data, ignore_index=True)
    return df


# ------------------------- Pipeline -----------------------------


def cleaning_pipeline(df, class_range):
    df = df.copy().reset_index(drop=True)
    # Old pickles have no source ID. A per-input-row identity prevents records
    # with missing LCCNs from being merged during place deduplication.
    if "source_record_id" not in df.columns:
        df["source_record_id"] = [f"legacy:{i}" for i in range(len(df))]
    else:
        missing_id = df["source_record_id"].isna()
        df.loc[missing_id, "source_record_id"] = [
            f"legacy:{i}" for i in df.index[missing_id]
        ]
    # For each record, filter to only target class_range
    # Drop those with no matching classification numbers
    df["matching_classifications"] = df["classifications"].map(
        lambda clist: filter_classifications(clist, class_range)
    )
    df = df[df["matching_classifications"].apply(lambda m: len(m) > 0)]

    # Take the *first* matching class (for numeric columns)
    df["target_classification"] = df["matching_classifications"].map(flatten_first)
    df["class_digits"] = df["target_classification"].map(
        lambda x: get_digits_for_class(x, class_range)
    )

    df["publisher_first"] = df["publishers"].map(flatten_first)
    df["personal_name_first"] = df["first_author"]
    df["title_first"] = df["title"]

    df["year_int_legacy"] = df["year"].map(get_years_ints)
    if "field_008" in df.columns:
        df["year_008"] = df["field_008"].map(_parse_date1_008)
    else:
        df["year_008"] = None
    df["publisher_year_int"] = df["publishers"].map(get_publishers_year_ints)
    selected_years = df.apply(_select_publication_year, axis=1)
    df["year_min"] = selected_years.map(lambda value: value[0])
    df["year_source"] = selected_years.map(lambda value: value[1])
    # Retain the historical column name for downstream compatibility, but it
    # now means the selected best publication-year estimate.
    df["year_int"] = df["year_min"]
    df["decade"] = df["year_min"].map(get_decade)
    df["publisher_clean"] = df["publisher_first"].map(clean_string)

    # Additional place-of-publication signal beyond 260/264 $a text (008
    # place code, 044 country codes, 752 hierarchical place name). Purely
    # additive/independent columns -- they do not feed places/places_clean/
    # city_group. Guarded on column presence for backward compatibility with
    # pickles produced before these fields were collected.
    if "field_008" in df.columns:
        df["place_code_008"] = df["field_008"].map(_parse_place_code_008)
        df["place_name_008"] = df["place_code_008"].map(decode_marc_country)

    if "country_codes_044" in df.columns:
        df["country_names_044"] = df["country_codes_044"].map(_decode_country_codes)

    if "place_hierarchy_752" in df.columns:
        df["place_752"] = df["place_hierarchy_752"].map(_flatten_place_hierarchy)

    # Explode on 'places' (each component city gets its own row). Compound
    # subfields like "Boston and New York" are split first so NYC is counted.
    if "places" in df.columns:
        df["places"] = df["places"].map(expand_places)
        df = df.explode("places").reset_index(drop=True)
    # Always compute these AFTER exploding
    df["places_clean"] = df["places"].map(clean_string)
    df["city_group"] = df["places_clean"].map(get_target_cities)

    # Drop rows with missing numeric classification digits or year before
    # deduplicating the normalized publication-place observations. Deduplicating
    # the raw exploded strings is insufficient: catalog records can repeat a
    # place as variants such as ``New York :`` and ``[New York] :``.
    df = df[df["class_digits"].notnull() & df["year_min"].notnull()]
    normalized_lccn = df["lccn"].astype("string").str.strip()
    has_lccn = normalized_lccn.notna() & normalized_lccn.ne("")
    df["record_dedup_id"] = df["source_record_id"].astype("string")
    df.loc[has_lccn, "record_dedup_id"] = "lccn:" + normalized_lccn[has_lccn]
    df = df.drop_duplicates(
        subset=["record_dedup_id", "places_clean", "year_min"], keep="first"
    ).reset_index(drop=True)

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean MARC LC records pickles into flat table."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with .pkl files")
    parser.add_argument("--output_csv", required=True, help="CSV file for output")
    parser.add_argument(
        "--output_pkl", default=None, help="Pickle file for output (optional)"
    )
    parser.add_argument(
        "--class_range",
        required=True,
        help="Classification prefix or range (e.g. PS or PR9000-PR9999)",
    )
    args = parser.parse_args()

    print(f"Loading record pickles from: {args.input_dir}")

    df = load_pickles_to_dataframe(args.input_dir)
    print(f"Loaded {len(df):,} records.")

    print(f"Filtering and cleaning for range: {args.class_range}")
    class_range = parse_range_spec(args.class_range)
    df_clean = cleaning_pipeline(df, class_range)
    print(
        f"""Remaining after cleaning/validity: {len(df_clean):,} records.
        Note that higher values possible due to exploded places."""
    )

    df_clean.to_csv(args.output_csv, index=False)
    print(f"Wrote cleaned CSV to: {args.output_csv}")

    if args.output_pkl:
        df_clean.to_pickle(args.output_pkl)
        print(f"Wrote cleaned DataFrame pickle to: {args.output_pkl}")
