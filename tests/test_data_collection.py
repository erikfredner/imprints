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


def test_matches_range_top_level_prefix_includes_subclasses():
    assert dc.matches_range("PR6053", "P")
    assert dc.matches_range("PS3553", "P")
    assert not dc.matches_range("QA76", "P")


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
    assert data["subtitle"] is None
    assert data["first_author"] == "Jewett, Sarah Orne,"
    assert data["places"] == ["Boston :"]


def test_process_record_non_matching_class_flagged_false():
    rec = make_record([("050", "0", [("a", "PR6019")])])
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["matches_class_range"] is False
    # Records are kept regardless of match (filtering happens in cleaning).
    assert data["classifications"] == ["PR6019"]


def test_process_record_extracts_subtitle_from_245_b():
    rec = make_record(
        [("245", "4", [("a", "The queen's twin /"), ("b", "and other stories /")])]
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["title"] == "The queen's twin /"
    assert data["subtitle"] == "and other stories /"


def test_process_record_subtitle_none_when_245_b_absent():
    rec = make_record([("245", "4", [("a", "The queen's twin /")])])
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["subtitle"] is None


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


def test_collect_subfields_captures_090_local_call_number():
    rec = make_record([("090", " ", [("a", "PS3555 .I123")])])
    buckets = dc.collect_subfields(rec)
    assert buckets[("090", "a")] == ["PS3555 .I123"]


def test_collect_subject_occurrences_groups_by_field_instance():
    rec = make_record(
        [
            ("650", "0", [("a", "American fiction"), ("x", "20th century")]),
            ("650", "0", [("a", "Detective and mystery stories"), ("x", "History")]),
        ]
    )
    occs = dc.collect_subject_occurrences(rec)
    assert len(occs) == 2
    assert occs[0]["subfields"] == [
        ("a", "American fiction"),
        ("x", "20th century"),
    ]
    assert occs[1]["subfields"] == [
        ("a", "Detective and mystery stories"),
        ("x", "History"),
    ]


def test_collect_subject_occurrences_ignores_non_subject_tags():
    rec = make_record([("700", " ", [("a", "Added entry")])])
    assert dc.collect_subject_occurrences(rec) == []


def test_collect_subject_occurrences_preserves_repeated_subfield_codes_within_occurrence():
    rec = make_record(
        [
            (
                "650",
                "0",
                [
                    ("a", "Hemingway, Ernest,"),
                    ("x", "Criticism and interpretation"),
                    ("x", "In literature"),
                ],
            )
        ]
    )
    occs = dc.collect_subject_occurrences(rec)
    assert len(occs) == 1
    assert occs[0]["subfields"] == [
        ("a", "Hemingway, Ernest,"),
        ("x", "Criticism and interpretation"),
        ("x", "In literature"),
    ]


def test_extract_008_returns_raw_text():
    rec = make_record(
        [("245", "0", [("a", "Title /")])],
        controlfields=[("008", "760729s1899    nyu           000 0 eng  ")],
    )
    assert dc.extract_008(rec) == "760729s1899    nyu           000 0 eng  "


def test_extract_008_returns_none_when_absent():
    rec = make_record([("245", "0", [("a", "Title /")])])
    assert dc.extract_008(rec) is None


def test_process_record_adds_new_fields_without_changing_existing_keys():
    rec = make_record(
        [
            ("010", " ", [("a", "   00000057 ")]),
            ("050", "0", [("a", "PS2132")]),
            ("090", " ", [("a", "PS2132 .L37")]),
            ("100", "1", [("a", "Jewett, Sarah Orne,"), ("d", "1849-1909.")]),
            ("245", "4", [("a", "The queen's twin /")]),
            ("264", "1", [("a", "Boston :"), ("b", "Houghton,"), ("c", "[1899]")]),
            ("600", "1", [("a", "Jewett, Sarah Orne,"), ("d", "1849-1909.")]),
            ("650", "0", [("a", "Short stories, American.")]),
        ],
        controlfields=[("008", "760729s1899    mau           000 0 eng  ")],
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))

    # Existing keys/values unchanged.
    assert data["lccn"] == "   00000057 "
    assert data["classifications"] == ["PS2132"]
    assert data["matches_class_range"] is True
    assert data["title"] == "The queen's twin /"
    assert data["year"] == ["[1899]"]
    assert data["places"] == ["Boston :"]
    assert data["publishers"] == ["Houghton,"]
    assert data["first_author"] == "Jewett, Sarah Orne,"

    # New keys present and correctly populated.
    assert data["local_call_numbers"] == ["PS2132 .L37"]
    assert data["first_author_dates"] == "1849-1909."
    assert data["field_008"] == "760729s1899    mau           000 0 eng  "
    assert [o["tag"] for o in data["subject_genre_fields"]] == ["600", "650"]
    assert data["subject_genre_fields"][0]["subfields"] == [
        ("a", "Jewett, Sarah Orne,"),
        ("d", "1849-1909."),
    ]


# ---------------------- 044 / 752 place extraction ----------------------


def test_collect_subfields_captures_044_country_codes():
    rec = make_record(
        [("044", " ", [("a", "nyu"), ("a", "enk"), ("c", "us"), ("c", "uk")])]
    )
    buckets = dc.collect_subfields(rec)
    assert buckets[("044", "a")] == ["nyu", "enk"]
    assert buckets[("044", "c")] == ["us", "uk"]


def test_process_record_no_044_yields_empty_lists():
    rec = make_record([("245", "0", [("a", "Title /")])])
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["country_codes_044"] == []
    assert data["country_codes_044_iso"] == []


def test_collect_place_occurrences_groups_by_field_instance():
    rec = make_record(
        [
            (
                "752",
                " ",
                [("a", "United States"), ("b", "New York (State)"), ("d", "New York")],
            ),
            ("752", " ", [("a", "England"), ("d", "London")]),
        ]
    )
    occs = dc.collect_place_occurrences(rec)
    assert len(occs) == 2
    assert occs[0]["subfields"] == [
        ("a", "United States"),
        ("b", "New York (State)"),
        ("d", "New York"),
    ]
    assert occs[1]["subfields"] == [("a", "England"), ("d", "London")]


def test_collect_place_occurrences_ignores_non_place_tags():
    rec = make_record([("700", " ", [("a", "Added entry")])])
    assert dc.collect_place_occurrences(rec) == []


def test_collect_place_occurrences_preserves_repeated_subfield_codes_within_occurrence():
    rec = make_record(
        [
            (
                "752",
                " ",
                [("a", "United States"), ("b", "New York (State)"), ("b", "Kings")],
            )
        ]
    )
    occs = dc.collect_place_occurrences(rec)
    assert len(occs) == 1
    assert occs[0]["subfields"] == [
        ("a", "United States"),
        ("b", "New York (State)"),
        ("b", "Kings"),
    ]


def test_process_record_no_752_yields_empty_list():
    rec = make_record([("245", "0", [("a", "Title /")])])
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["place_hierarchy_752"] == []


def test_process_record_captures_044_and_752_together():
    rec = make_record(
        [
            ("050", "0", [("a", "PS2132")]),
            ("044", " ", [("a", "nyu"), ("c", "us")]),
            (
                "752",
                " ",
                [("a", "United States"), ("b", "New York (State)"), ("d", "New York")],
            ),
            ("752", " ", [("a", "England"), ("d", "London")]),
            ("264", "1", [("a", "Boston :"), ("c", "[1899]")]),
        ]
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))

    assert data["country_codes_044"] == ["nyu"]
    assert data["country_codes_044_iso"] == ["us"]
    assert len(data["place_hierarchy_752"]) == 2
    assert data["place_hierarchy_752"][0]["subfields"] == [
        ("a", "United States"),
        ("b", "New York (State)"),
        ("d", "New York"),
    ]
    assert data["place_hierarchy_752"][1]["subfields"] == [
        ("a", "England"),
        ("d", "London"),
    ]
    # Pre-existing keys unaffected.
    assert data["places"] == ["Boston :"]
    assert data["year"] == ["[1899]"]


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
