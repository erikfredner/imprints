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


def test_get_year_int_rejects_non_year_digit_runs():
    # Street numbers and other implausible values are not years.
    assert cl.get_year_int("1420 Chestnut Street") is None
    assert cl.get_year_int("8728") is None
    # A run embedded in a longer digit string (e.g. a zip code) is never a
    # year -- "New York, 10014" must not yield 1001.
    assert cl.get_year_int("New York, 10014") is None
    # A plausible year is still found even after implausible runs.
    assert cl.get_year_int("Suite 8100, est. 1852") == 1852


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
    assert cl.get_digits_for_class("PS3555.123", cl.parse_range_spec("P")) == 3555


# ---------------------- classification matching ----------------------


def test_matches_range_prefix_fix_mirrors_collection():
    assert cl.matches_range("PS3553", "PS")
    assert not cl.matches_range("PSA123", "PS")
    assert not cl.matches_range("PZ3.J55", "PS")
    assert cl.matches_range("PR6053", "P")
    assert cl.matches_range("PS3553", "P")
    assert not cl.matches_range("QA76", "P")


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
    assert cl.split_places("Boston and Chicago") == ["Boston", "Chicago"]
    assert cl.split_places("New York; London") == ["New York", "London"]
    assert cl.split_places("New York & London") == ["New York", "London"]
    assert cl.split_places("Philadelphia & London,") == ["Philadelphia", "London,"]
    # Square brackets mark supplied MARC text, not a single grouped place.
    assert cl.split_places("[New York and London]") == ["New York", "London"]
    assert cl.split_places("[Harmondsworth; Baltimore]") == [
        "Harmondsworth",
        "Baltimore",
    ]
    # Conjunctions within one place/address must not manufacture extra rows.
    assert cl.split_places("[Trinidad and Tobago, West Indies?]") == [
        "[Trinidad and Tobago, West Indies?]"
    ]
    assert cl.split_places("[Antigua and Barbuda]") == ["[Antigua and Barbuda]"]
    assert cl.split_places("New York (Fourth Avenue and Twenty-third Street)") == [
        "New York (Fourth Avenue and Twenty-third Street)"
    ]
    assert cl.split_places(
        "New York (Fourth Avenue and Twenty-third Street) and London"
    ) == ["New York (Fourth Avenue and Twenty-third Street)", "London"]
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


def test_expand_places_bracketed_city_list_recovers_nyc():
    places = cl.expand_places("[New York and London]")
    assert places == ["New York", "London"]
    assert [cl.get_target_cities(place) for place in places] == [
        "New York City",
        "Other",
    ]


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


def test_cleaning_pipeline_top_level_class_and_single_place_conjunctions():
    df = pd.DataFrame(
        [
            _record(
                ["PS3553"],
                ["[Trinidad and Tobago, West Indies?]"],
                ["1980"],
            ),
            _record(
                ["PR6053"],
                ["New York (Fourth Avenue and Twenty-third Street)"],
                ["1980"],
            ),
        ]
    )

    out = cl.cleaning_pipeline(df, cl.parse_range_spec("P"))

    assert out["target_classification"].tolist() == ["PS3553", "PR6053"]
    assert out["class_digits"].tolist() == [3553, 6053]
    assert len(out) == 2


# ---------------------- 008 / 044 / 752 place extraction ----------------------


def test_parse_place_code_008_extracts_bytes_15_17():
    assert cl._parse_place_code_008("760729s1899    nyu           000 0 eng  ") == "nyu"


def test_parse_place_code_008_guards_missing_or_short():
    assert cl._parse_place_code_008(None) is None
    assert cl._parse_place_code_008("") is None
    assert cl._parse_place_code_008("760729s1899") is None  # too short


def test_parse_place_code_008_guards_pandas_nan():
    # Regression: a DataFrame column mixing present and missing field_008
    # values stores the missing ones as float NaN (pandas' modern string
    # dtype), not Python None -- `not field_008` doesn't catch that, and
    # `len(nan)` raises. Single-value/single-row calls with a bare None
    # don't reproduce this; a multi-row, mixed-presence column does.
    s = pd.Series([None, "760729s1899    nyu           000 0 eng  "], name="field_008")
    out = s.map(cl._parse_place_code_008)
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == "nyu"


def test_decode_country_codes_order_preserving_with_unknown_gaps():
    assert cl._decode_country_codes(["nyu", "zz9", "enk"]) == [
        "New York (State)",
        None,
        "England",
    ]
    assert cl._decode_country_codes(None) is None
    assert cl._decode_country_codes([]) == []


def test_flatten_place_hierarchy_joins_in_fixed_order():
    occurrences = [
        {
            "tag": "752",
            "ind2": " ",
            "subfields": [
                ("d", "New York"),
                ("a", "United States"),
                ("b", "New York (State)"),
            ],
        },
        {"tag": "752", "ind2": " ", "subfields": [("a", "England"), ("d", "London")]},
        {"tag": "752", "ind2": " ", "subfields": []},  # no usable subfields
    ]
    assert cl._flatten_place_hierarchy(occurrences) == [
        "United States, New York (State), New York",
        "England, London",
    ]
    assert cl._flatten_place_hierarchy([]) is None
    assert cl._flatten_place_hierarchy(None) is None


def _record_with_marc_fields(
    classifications,
    places,
    year,
    field_008=None,
    country_codes_044=None,
    place_hierarchy_752=None,
):
    rec = _record(classifications, places, year)
    rec["field_008"] = field_008
    rec["country_codes_044"] = country_codes_044 or []
    rec["place_hierarchy_752"] = place_hierarchy_752 or []
    return rec


def test_cleaning_pipeline_decodes_008_and_044():
    df = pd.DataFrame(
        [
            _record_with_marc_fields(
                ["PS3553"],
                ["New York :"],
                ["1980"],
                field_008="760729s1899    nyu           000 0 eng  ",
                country_codes_044=["nyu", "enk"],
            )
        ]
    )
    out = cl.cleaning_pipeline(df, cl.parse_range_spec("PS"))
    assert (out["place_code_008"] == "nyu").all()
    assert (out["place_name_008"] == "New York (State)").all()
    assert out["country_names_044"].iloc[0] == ["New York (State)", "England"]


def test_cleaning_pipeline_flattens_752():
    df = pd.DataFrame(
        [
            _record_with_marc_fields(
                ["PS3553"],
                ["New York :"],
                ["1980"],
                place_hierarchy_752=[
                    {
                        "tag": "752",
                        "ind2": " ",
                        "subfields": [
                            ("a", "United States"),
                            ("b", "New York (State)"),
                            ("d", "New York"),
                        ],
                    }
                ],
            )
        ]
    )
    out = cl.cleaning_pipeline(df, cl.parse_range_spec("PS"))
    assert out["place_752"].iloc[0] == ["United States, New York (State), New York"]


def test_cleaning_pipeline_new_fields_do_not_affect_city_group():
    # Capture-only scope: 008/044/752 must not change places/places_clean/
    # city_group, even alongside a compound places value.
    df = pd.DataFrame(
        [
            _record_with_marc_fields(
                ["PS3553"],
                ["Boston and New York :"],
                ["1980"],
                field_008="760729s1899    mau           000 0 eng  ",
                country_codes_044=["mau"],
            )
        ]
    )
    out = cl.cleaning_pipeline(df, cl.parse_range_spec("PS"))
    groups = sorted(out["city_group"].tolist())
    assert groups == ["New York City", "Other"]
    assert sorted(out["places"].tolist()) == ["Boston", "New York :"]


def test_cleaning_pipeline_handles_mixed_missing_and_present_field_008():
    # Regression: a multi-row DataFrame where some records have field_008
    # and others don't must not crash cleaning_pipeline (see
    # test_parse_place_code_008_guards_pandas_nan).
    df = pd.DataFrame(
        [
            _record_with_marc_fields(["PS1"], ["New York :"], ["1980"]),
            _record_with_marc_fields(
                ["PS2"],
                ["Boston :"],
                ["1980"],
                field_008="760729s1899    mau           000 0 eng  ",
            ),
        ]
    )
    out = cl.cleaning_pipeline(df, cl.parse_range_spec("PS"))
    assert pd.isna(out["place_code_008"].iloc[0])
    assert pd.isna(out["place_name_008"].iloc[0])
    assert out["place_code_008"].iloc[1] == "mau"
    assert out["place_name_008"].iloc[1] == "Massachusetts"
