"""Regression tests for extraction/cleaning accuracy improvements."""

import pandas as pd
import pytest

from conftest import make_record
from imprints import data_cleaning as cl
from imprints import data_collection as dc


def _record(lccn=None, source_record_id=None, **overrides):
    record = {
        "lccn": lccn,
        "source_record_id": source_record_id,
        "classifications": ["PS3553"],
        "title": "Title",
        "year": ["1980"],
        "places": ["Boston"],
        "publishers": ["Publisher"],
        "first_author": "Author",
    }
    record.update(overrides)
    return record


def test_imprint_provenance_prefers_publication_over_earlier_copyright():
    marc = make_record(
        [
            ("050", "0", [("a", "PS3553")]),
            ("264", "1", [("a", "Boston"), ("c", "2001")]),
            ("264", "4", [("c", "©1958")]),
        ]
    )
    extracted = dc.process_record(marc, dc.parse_range_spec("PS"), "file:1")
    out = cl.cleaning_pipeline(pd.DataFrame([extracted]), cl.parse_range_spec("PS"))
    assert out.loc[0, "year_min"] == 2001
    assert out.loc[0, "year_source"] == "260/264 publication"


def test_008_date_precedes_flattened_legacy_and_publisher_years():
    record = _record(
        field_008="760729s2001    mau           000 0 eng  ",
        year=["©1958"],
        publishers=["Founded 1890"],
    )
    out = cl.cleaning_pipeline(pd.DataFrame([record]), cl.parse_range_spec("PS"))
    assert out.loc[0, "year_min"] == 2001
    assert out.loc[0, "year_source"] == "008 date1"


def test_questionable_008_range_is_not_treated_as_exact_lower_bound():
    assert cl._parse_date1_008("900217q19001989wiua          000 0 eng  ") is None
    assert cl._parse_date1_008("900217q19801980wiua          000 0 eng  ") == 1980


def test_broad_multiple_date_008_is_not_treated_as_exact_lower_bound():
    assert cl._parse_date1_008("770303m19001999enk           00011 eng  ") is None
    assert cl._parse_date1_008("770303m19611963enk           00011 eng  ") == 1961


def test_cataloger_ie_correction_supersedes_erroneous_printed_year():
    assert cl.get_year_int("c1863 [i.e. 1963]") == 1963
    assert cl.get_year_int("[1944, i.e. 1994]") == 1994


def test_broad_uncertain_publication_range_is_not_an_exact_year():
    assert cl.get_year_int("[1900-1993]") is None
    assert cl.get_year_int("2000-2005") == 2000


def test_distant_years_in_prose_are_not_mistaken_for_a_range():
    assert cl.get_year_int("Published 1890; renewed 1950") == 1890
    assert cl.get_year_int("First published 1940; this edition 1980") == 1940


def test_missing_lccn_records_are_not_merged():
    records = [_record(), _record(title="A different title")]
    out = cl.cleaning_pipeline(pd.DataFrame(records), cl.parse_range_spec("PS"))
    assert len(out) == 2
    assert out["source_record_id"].nunique() == 2


def test_duplicate_identified_records_are_merged_by_normalized_lccn():
    records = [
        _record(lccn=" 123 ", source_record_id="file:1"),
        _record(lccn="123", source_record_id="file:2"),
    ]
    out = cl.cleaning_pipeline(pd.DataFrame(records), cl.parse_range_spec("PS"))
    assert len(out) == 1
    assert out.loc[0, "record_dedup_id"] == "lccn:123"


def test_unicode_place_cleaning_preserves_letters():
    assert cl.clean_string("São Paulo") == "sao paulo"
    assert cl.clean_string("Montréal") == "montreal"
    assert cl.clean_string("北京") == "北京"


def test_repeated_752_components_are_preserved():
    occurrences = [{"tag": "752", "ind2": " ", "subfields": [
        ("a", "United States"), ("b", "New York"), ("b", "Kings"),
        ("d", "Brooklyn"),
    ]}]
    assert cl._flatten_place_hierarchy(occurrences) == [
        "United States, New York, Kings, Brooklyn"
    ]


@pytest.mark.parametrize("module", [dc, cl])
def test_mismatched_range_prefix_is_rejected(module):
    with pytest.raises(ValueError, match="same LC prefix"):
        module.parse_range_spec("PR9000-PS9999")
