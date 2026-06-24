"""Tests for MARC extraction (imprints.data_collection)."""

from conftest import make_record

from imprints import data_collection as dc


# ---------------------- class-range parsing ----------------------


def test_parse_class_prefix_and_digits():
    assert dc.parse_class("PS3555.123") == ("PS", 3555)
    assert dc.parse_class("PS") == ("PS", None)
    assert dc.parse_class("813.49") == (None, None)
    assert dc.parse_class("") == (None, None)


def test_parse_range_spec_forms():
    assert dc.parse_range_spec("PS") == {"prefix": "PS", "min": None, "max": None}
    assert dc.parse_range_spec("PR9000-PR9999") == {
        "prefix": "PR",
        "min": 9000,
        "max": 9999,
    }


def test_matches_range_basic():
    assert dc.matches_range("PS3553", "PS")
    assert not dc.matches_range("PZ3.J55", "PS")
    assert dc.matches_range("PR9053", "PR", 9000, 9999)
    assert not dc.matches_range("PR8000", "PR", 9000, 9999)


def test_matches_range_rejects_longer_prefix():
    # Regression: "PSA123" must NOT match prefix "PS" (was a false positive
    # when the check used startswith instead of exact alpha-class equality).
    assert not dc.matches_range("PSA123", "PS")
    assert not dc.matches_range("PST", "PS")


def test_matches_range_accepts_list():
    assert dc.matches_range(["PZ3.J55", "PS2132"], "PS")
    assert not dc.matches_range(["PZ3.J55", "PR1234"], "PS")


def test_filter_classification():
    cr = dc.parse_range_spec("PS")
    assert dc.filter_classification(["PS2132", "PZ3.J55"], cr)
    assert not dc.filter_classification(["PZ3.J55"], cr)
    assert not dc.filter_classification([], cr)


# ---------------------- single-pass extraction ----------------------


def test_collect_subfields_buckets_by_tag_and_code():
    rec = make_record(
        [
            ("010", " ", [("a", "  00000057 ")]),
            ("050", "0", [("a", "PS2132"), ("b", ".Q4"), ("a", "PZ3.J55")]),
            ("245", "0", [("a", "The queen's twin /"), ("c", "by ...")]),
        ]
    )
    buckets = dc.collect_subfields(rec)
    assert buckets[("050", "a")] == ["PS2132", "PZ3.J55"]
    assert buckets[("245", "a")] == ["The queen's twin /"]
    # Datafields outside the allowlist are never collected.
    rec2 = make_record([("700", " ", [("a", "Added entry")])])
    assert ("700", "a") not in dc.collect_subfields(rec2)


def test_collect_subfields_skips_empty_text():
    rec = make_record([("264", "1", [("a", None), ("a", "   "), ("a", "New York :")])])
    buckets = dc.collect_subfields(rec)
    assert buckets[("264", "a")] == ["New York :"]


def test_264_indicator_filters_place_and_publisher():
    # 264 _1 = publication (kept), 264 _3 = manufacture (printer; dropped),
    # 264 _4 = copyright (place dropped, but year kept as pub-date proxy).
    rec = make_record(
        [
            (
                "264",
                "1",
                [("a", "Boston and New York :"), ("b", "Houghton,"), ("c", "[1899]")],
            ),
            (
                "264",
                "3",
                [("a", "Cambridge :"), ("b", "Riverside Press,"), ("c", "1899.")],
            ),
            ("264", "4", [("c", "©1898")]),
        ]
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["places"] == ["Boston and New York :"]
    assert data["publishers"] == ["Houghton,"]
    # Cambridge (manufacture) excluded; all $c years retained for year_min.
    assert "Cambridge :" not in data["places"]
    assert set(data["year"]) == {"[1899]", "1899.", "©1898"}


def test_process_record_full_record():
    rec = make_record(
        [
            ("010", " ", [("a", "   00000057 ")]),
            ("050", "0", [("a", "PS2132"), ("a", "PZ3.J55")]),
            ("100", "1", [("a", "Jewett, Sarah Orne,")]),
            ("245", "4", [("a", "The queen's twin /")]),
            ("264", "1", [("a", "Boston :"), ("b", "Houghton,"), ("c", "[1899]")]),
        ]
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["lccn"] == "   00000057 "
    assert data["classifications"] == ["PS2132", "PZ3.J55"]
    assert data["matches_class_range"] is True
    assert data["title"] == "The queen's twin /"
    assert data["first_author"] == "Jewett, Sarah Orne,"
    assert data["places"] == ["Boston :"]


def test_process_record_non_matching_class_flagged_false():
    rec = make_record([("050", "0", [("a", "PR6019")])])
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["matches_class_range"] is False
    # Records are kept regardless of match (filtering happens in cleaning).
    assert data["classifications"] == ["PR6019"]


def test_260_and_264_merge_order_preserving_dedup():
    rec = make_record(
        [
            ("260", " ", [("a", "New York :"), ("c", "1950")]),
            ("264", "1", [("a", "New York :"), ("c", "1951")]),
        ]
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["places"] == ["New York :"]  # deduped
    assert data["year"] == ["1950", "1951"]


# ---------------------- iterparse round-trip ----------------------


def test_parse_records_round_trip(tmp_path, marc_gz_writer):
    records = [
        [("050", "0", [("a", "PS3553")]), ("264", "1", [("a", "New York :")])],
        [("050", "0", [("a", "PR6019")]), ("264", "1", [("a", "London :")])],
        [("050", "0", [("a", "PS1")]), ("264", "1", [("a", "Boston :")])],
    ]
    path = marc_gz_writer(tmp_path / "sample.xml.gz", records)
    parsed = [
        dc.process_record(rec, dc.parse_range_spec("PS"))
        for rec in dc.parse_records(path)
    ]
    assert len(parsed) == 3
    assert [p["matches_class_range"] for p in parsed] == [True, False, True]


def test_parse_records_drops_processed_siblings(tmp_path, marc_gz_writer):
    # The memory fix deletes already-processed <record> nodes from the tree.
    # (For a tiny in-memory file lxml buffers the whole document up front, so
    # we observe the tree *shrinking* as records are consumed rather than
    # staying tiny; on a streamed multi-GB file only a chunk is ever resident.)
    # Without the fix the parent would stay at its full size throughout.
    records = [[("050", "0", [("a", f"PS{i}")])] for i in range(20)]
    path = marc_gz_writer(tmp_path / "many.xml.gz", records)
    sizes = [
        len(elem.getparent()) if elem.getparent() is not None else 0
        for elem in dc.parse_records(path)
    ]
    assert sizes == sorted(sizes, reverse=True)  # non-increasing: siblings dropped
    assert sizes[-1] <= 2  # ends bounded, not at the original 20
