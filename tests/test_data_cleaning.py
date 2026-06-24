"""Tests for normalization/cleaning (imprints.data_cleaning)."""

import numpy as np
import pandas as pd

from imprints import data_cleaning as cl


# ---------------------- string + year helpers ----------------------


def test_clean_string():
    assert cl.clean_string("New York, N.Y.") == "new york n y"
    assert cl.clean_string("  Brooklyn!! ") == "brooklyn"
    assert cl.clean_string(None) is None
    assert cl.clean_string(123) == ""  # digits stripped


def test_get_year_int_forms():
    assert cl.get_year_int("[1899]") == 1899
    assert cl.get_year_int("1899.") == 1899
    assert cl.get_year_int("©1899") == 1899
    assert cl.get_year_int("c1995") == 1995
    assert cl.get_year_int("2000-2005") == 2000  # first 4-digit run
    # Unparseable -> None (such rows are dropped downstream when no other year).
    assert cl.get_year_int("[19--]") is None
    assert cl.get_year_int("18th century") is None
    assert cl.get_year_int(None) is None


def test_get_years_ints_collection_and_scalar():
    assert cl.get_years_ints(["c1995", "[1990]", "n.d."]) == 1990
    assert cl.get_years_ints([]) is None
    assert cl.get_years_ints(None) is None
    assert cl.get_years_ints(np.array(["1980", "1975"])) == 1975
    assert cl.get_years_ints("1962") == 1962


def test_get_publishers_year_ints():
    assert cl.get_publishers_year_ints(["Knopf, c1981", "1979"]) == 1979
    assert cl.get_publishers_year_ints([]) is None
    assert cl.get_publishers_year_ints(None) is None


def test_get_decade():
    assert cl.get_decade(1983) == 1980
    assert cl.get_decade("1999") == 1990
    assert cl.get_decade(None) is None
    assert cl.get_decade("not a year") is None


def test_get_digits_for_class():
    cr = cl.parse_range_spec("PS")
    assert cl.get_digits_for_class("PS3555.123", cr) == 3555
    assert cl.get_digits_for_class("PZ3.J55", cr) is None
    assert cl.get_digits_for_class(None, cr) is None


# ---------------------- classification matching ----------------------


def test_matches_range_prefix_fix_mirrors_collection():
    assert cl.matches_range("PS3553", "PS")
    assert not cl.matches_range("PSA123", "PS")
    assert not cl.matches_range("PZ3.J55", "PS")


def test_filter_classifications():
    cr = cl.parse_range_spec("PS")
    assert cl.filter_classifications(["PS2132", "PZ3.J55"], cr) == ["PS2132"]
    assert cl.filter_classifications(["PZ3.J55"], cr) == []
    assert cl.filter_classifications("PS1", cr) == ["PS1"]
    assert cl.filter_classifications(None, cr) == []


# ---------------------- place handling ----------------------


def test_flatten_first():
    assert cl.flatten_first(["a", "b"]) == "a"
    assert cl.flatten_first([]) is None
    assert cl.flatten_first("x") == "x"


def test_normalize_places():
    assert cl.normalize_places(["New York"]) == ["New York"]
    assert cl.normalize_places([]) == [None]
    assert cl.normalize_places(None) == [None]
    assert cl.normalize_places("Boston") == ["Boston"]


def test_split_places_compound():
    assert cl.split_places("Boston and New York :") == ["Boston", "New York :"]
    assert cl.split_places("New York; London") == ["New York", "London"]
    assert cl.split_places("New York & London") == ["New York", "London"]
    # Comma is NOT a separator (qualifies a single place).
    assert cl.split_places("Flushing, NY") == ["Flushing, NY"]
    # "and" inside a word must not split.
    assert cl.split_places("Portland") == ["Portland"]
    assert cl.split_places(None) == [None]


def test_expand_places_dedup_and_missing():
    assert cl.expand_places(["Boston and New York", "New York"]) == [
        "Boston",
        "New York",
    ]
    assert cl.expand_places([]) == [None]
    assert cl.expand_places(None) == [None]


def test_get_target_cities():
    assert cl.get_target_cities("New York :") == "New York City"
    assert cl.get_target_cities("Brooklyn, N.Y.") == "New York City"
    assert cl.get_target_cities("Boston") == "Other"
    assert cl.get_target_cities("") == "No place of publication"
    assert cl.get_target_cities(None) == "No place of publication"
    # The compound string itself is "Other"; recovery happens via split_places.
    assert cl.get_target_cities("Boston and New York") == "Other"


# ---------------------- end-to-end pipeline ----------------------


def _record(classifications, places, year, publishers=("Pub, 1980",)):
    return {
        "lccn": "123",
        "classifications": classifications,
        "title": "T",
        "year": year,
        "places": places,
        "publishers": list(publishers),
        "first_author": "A",
    }


def test_cleaning_pipeline_filters_explodes_and_recovers_nyc():
    df = pd.DataFrame(
        [
            # PS record with a compound place -> NYC must be recovered.
            _record(["PS3553"], ["Boston and New York :"], ["1980"]),
            # Non-PS record is dropped entirely.
            _record(["PR6019"], ["London :"], ["1980"]),
            # PS record with no parseable year anywhere -> dropped.
            _record(["PS1"], ["New York :"], ["[19--]"], publishers=[]),
        ]
    )
    out = cl.cleaning_pipeline(df, cl.parse_range_spec("PS"))

    # PR row gone; the no-year PS row gone.
    assert set(out["target_classification"]) == {"PS3553"}
    # Compound place exploded into two rows: Boston (Other) + New York (NYC).
    groups = sorted(out["city_group"].tolist())
    assert groups == ["New York City", "Other"]
    assert (out["year_min"] == 1980).all()
    assert (out["class_digits"] == 3553).all()


def test_cleaning_pipeline_year_min_uses_publisher_year():
    # No year in 'year', but publisher string carries one.
    df = pd.DataFrame([_record(["PS3553"], ["New York :"], [])])
    out = cl.cleaning_pipeline(df, cl.parse_range_spec("PS"))
    assert (out["year_min"] == 1980).all()
