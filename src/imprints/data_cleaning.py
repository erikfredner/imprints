import os
import pandas as pd
import pickle
import re
from functools import lru_cache

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


def matches_range(classification, prefix, num_min=None, num_max=None):
    """
    Test if a classification matches a given prefix and optional num range.
    """
    if not classification or not isinstance(classification, str):
        return False
    cls, num = parse_class(classification.strip())
    if not cls or not classification.strip().startswith(prefix):
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
    m = re.match(rf"^{prefix}(\d+)", s)
    return int(m.group(1)) if m else None


def get_year_int(year):
    """Extract a four-digit year from a string (or return None)."""
    if year is None or (isinstance(year, float) and pd.isna(year)):
        return None
    match = re.search(r"\d{4}", str(year))
    return int(match.group()) if match else None


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
    except:
        return None


def clean_string(s):
    """Lowercase, replace punctuation with spaces, collapse to single spaces."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    s = str(s)
    s = re.sub(r"[^a-zA-Z]", " ", s)
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

    df["year_int"] = df["year"].map(get_years_ints)
    df["publisher_year_int"] = df["publishers"].map(get_publishers_year_ints)
    df["year_min"] = df[["year_int", "publisher_year_int"]].min(axis=1)
    df["decade"] = df["year_min"].map(get_decade)
    df["publisher_clean"] = df["publisher_first"].map(clean_string)

    # Explode on 'places' (each place gets its own row)
    if "places" in df.columns:
        df["places"] = df["places"].map(normalize_places)
        df = df.explode("places").reset_index(drop=True)
    # Always compute these AFTER exploding
    df["places_clean"] = df["places"].map(clean_string)
    df["city_group"] = df["places_clean"].map(get_target_cities)

    # Optionally drop any rows with missing numeric classification digits or year
    df = df[df["class_digits"].notnull() & df["year_min"].notnull()]

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
