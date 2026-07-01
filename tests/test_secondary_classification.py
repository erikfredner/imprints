"""Tests for imprints.secondary_classification."""

import pandas as pd
from conftest import make_record

from imprints import data_collection as dc
from imprints import secondary_classification as sc


def _occ(tag, subfields, ind2="0"):
    return {"tag": tag, "ind2": ind2, "subfields": subfields}


def _base_record(**overrides):
    record = {
        "lccn": "12345",
        "classifications": ["PS3500"],
        "local_call_numbers": [],
        "first_author": None,
        "first_author_dates": None,
        "subject_genre_fields": [],
        "field_008": None,
    }
    record.update(overrides)
    return record


# ---------------------- Step 1: scope filter ----------------------


def test_in_ps_scope_matches_050_with_and_without_whitespace():
    assert sc.in_ps_scope(_base_record(classifications=["PS3500"]))
    assert sc.in_ps_scope(_base_record(classifications=["PS 3500"]))
    assert not sc.in_ps_scope(_base_record(classifications=["PR3500"]))


def test_in_ps_scope_falls_back_to_090_when_050_absent():
    record = _base_record(classifications=[], local_call_numbers=["PS3500 .B4"])
    assert sc.in_ps_scope(record)


def test_in_ps_scope_does_not_fallback_when_050_present_but_non_ps():
    record = _base_record(classifications=["PR6019"], local_call_numbers=["PS3500 .B4"])
    assert not sc.in_ps_scope(record)


def test_in_ps_scope_excludes_when_neither_matches():
    record = _base_record(classifications=[], local_call_numbers=["PR6019"])
    assert not sc.in_ps_scope(record)


# ---------------------- name_key / self_subject ----------------------


def test_name_key_normalizes_punctuation_and_case():
    a = sc._name_key("Hemingway, Ernest,", "1899-1961.")
    b = sc._name_key("HEMINGWAY, ERNEST", "1899-1961")
    assert a == b


def test_name_key_compares_birth_year_not_full_date_string():
    a = sc._name_key("Hemingway, Ernest,", "1899-")
    b = sc._name_key("Hemingway, Ernest,", "1899-1961.")
    assert a == b


def test_name_key_none_when_no_name():
    assert sc._name_key(None, "1899-1961.") is None
    assert sc._name_key("", "1899-1961.") is None


# ---------------------- R1 ----------------------


def test_r1_fires_when_600_subject_differs_from_creator():
    record = _base_record(
        first_author="Hemingway, Ernest,",
        first_author_dates="1899-1961.",
        subject_genre_fields=[
            _occ("600", [("a", "Fitzgerald, F. Scott,"), ("d", "1896-1940.")])
        ],
    )
    result = sc.classify_record(record)
    assert "R1" in result["matched_rules"]
    assert result["is_secondary"] is True
    assert result["confidence"] == "high"


def test_r1_does_not_fire_when_only_600_is_self_subject():
    record = _base_record(
        first_author="Hemingway, Ernest,",
        first_author_dates="1899-1961.",
        subject_genre_fields=[
            _occ("600", [("a", "Hemingway, Ernest,"), ("d", "1899-1961.")])
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["self_subject"] is True


def test_r1_fires_when_no_100_present():
    # Festschrift / edited critical anthology with no single creator.
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[_occ("600", [("a", "Faulkner, William,")])],
    )
    result = sc.classify_record(record)
    assert "R1" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_r1_does_not_fire_when_600_has_genre_instantiation_subdivision():
    # A figure appearing AS A CHARACTER in a primary work ("Jesus Christ --
    # Fiction.") is not the same as a work ABOUT that figure -- the bare
    # name alone can't distinguish these, only the subdivision can.
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[_occ("600", [("a", "Jesus Christ"), ("x", "Fiction.")])],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert result["is_secondary"] is False


def test_r1_does_not_fire_on_fictitious_character_with_genre_instantiation():
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ(
                "600",
                [
                    ("a", "Uncle Tom"),
                    ("c", "(Fictitious character)"),
                    ("x", "Fiction."),
                ],
            )
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert result["is_secondary"] is False


def test_r1_fires_on_bare_name_600_with_no_subdivision():
    # Contrast with the two tests above: a bare name with no subdivision at
    # all is the actual signature of "this book discusses that person as a
    # literary figure" (e.g. a multi-author critical survey where each
    # subject gets an unadorned 600).
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ("600", [("a", "Bryant, William Cullen,"), ("d", "1794-1878.")])
        ],
    )
    result = sc.classify_record(record)
    assert "R1" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_r1_fires_from_610_and_630_not_only_600():
    # Corporate name (610) and uniform title (630) subjects should trigger
    # the same bare-name-as-subject signal as 600, per spec Step 3 rule 5.
    for tag in ("610", "611", "630"):
        record = _base_record(
            first_author=None,
            first_author_dates=None,
            subject_genre_fields=[_occ(tag, [("a", "Some Corporate or Title Name")])],
        )
        result = sc.classify_record(record)
        assert "R1" in result["matched_rules"], tag
        assert result["is_secondary"] is True, tag


def test_r1_skips_non_lcsh_bare_duplicate_of_lcsh_subdivided_sibling():
    # FAST (and other non-LCSH vocabularies) often duplicate an
    # LCSH-subdivided heading as a separate bare heading with the
    # subdivision stripped off, e.g. LCSH "610 Templars -- Fiction." plus a
    # FAST sibling "610 Templars. $2 fast" with ind2 "7" and no subdivision.
    # R1 must be restricted to ind2 "0" (LCSH) so the bare FAST duplicate
    # doesn't fire R1 on a record already correctly resolved as primary by
    # its LCSH sibling (R6).
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ("610", [("a", "Templars"), ("x", "Fiction.")], ind2="0"),
            _occ("610", [("a", "Templars."), ("2", "fast")], ind2="7"),
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert "R6" in result["matched_rules"]
    assert result["is_secondary"] is False


def test_r1_does_not_fire_when_occurrence_also_has_criticism_subdivision():
    # A 600 with an explicit "Criticism and interpretation" subdivision is
    # explained by R2 -- it isn't a "bare name" case, so R1 shouldn't also
    # fire redundantly for the same occurrence.
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ(
                "600",
                [
                    ("a", "Bryant, William Cullen,"),
                    ("x", "Criticism and interpretation"),
                ],
            )
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert "R2" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_ambiguous_character_qualifier_with_no_subdivision_sets_review_flag_only():
    # LC uses the identical bare heading ("Robin Hood (Legendary character)")
    # both for a work in which the character appears and for criticism about
    # the character/legend -- unresolvable from the heading alone, so no
    # rule should fire, but review_flag must be set.
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ("600", [("a", "Robin Hood"), ("c", "(Legendary character)")])
        ],
    )
    result = sc.classify_record(record)
    assert result["matched_rules"] == []
    assert result["is_secondary"] is False
    assert result["review_flag"] is True


def test_ambiguous_character_qualifier_without_parens_or_in_a():
    # The qualifier can render without parentheses and outside $a (e.g. in
    # $c), depending on the source record -- CHARACTER_QUALIFIERS must match
    # the full concatenated occurrence text, not just $a/$c with parens.
    record = _base_record(
        subject_genre_fields=[
            _occ("600", [("a", "Hawke, Alex,"), ("c", "Fictitious character.")])
        ]
    )
    result = sc.classify_record(record)
    assert result["matched_rules"] == []
    assert result["is_secondary"] is False
    assert result["review_flag"] is True


def test_r6_genre_instantiation_terms_include_juvenile_literature_and_literary_collections():
    for phrase in ("Juvenile literature.", "Literary collections."):
        record = _base_record(
            subject_genre_fields=[
                _occ("650", [("a", "American fiction"), ("x", phrase)])
            ]
        )
        result = sc.classify_record(record)
        assert "R6" in result["matched_rules"], phrase
        assert result["is_secondary"] is False, phrase


def test_r6_fires_from_600_genre_instantiation_not_only_650():
    # R6 previously only looked at 650 -- it must fire from any subject/genre
    # tag now, since that's the same signal R1 needs generalized above.
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[_occ("600", [("a", "Jesus Christ"), ("x", "Fiction.")])],
    )
    result = sc.classify_record(record)
    assert "R6" in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["confidence"] == "high"


# ---------------------- R2 / R3 / R5 / R9 ----------------------


def test_r2_fires_on_criticism_substring_in_650_x():
    record = _base_record(
        subject_genre_fields=[
            _occ("650", [("a", "American fiction"), ("x", "History and criticism")])
        ]
    )
    result = sc.classify_record(record)
    assert "R2" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_r3_fires_on_in_literature_substring():
    record = _base_record(
        subject_genre_fields=[
            _occ("600", [("a", "Lincoln, Abraham,"), ("x", "In literature")])
        ]
    )
    result = sc.classify_record(record)
    assert "R3" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_655_occurrences_produce_no_signal():
    # R4 (655 genre/form term rule) was removed from the spec: 655 is
    # extracted upstream (data_collection.SUBJECT_GENRE_TAGS) but no longer
    # consumed by any classification rule here.
    record = _base_record(
        subject_genre_fields=[_occ("655", [("a", "Literary criticism.")])]
    )
    result = sc.classify_record(record)
    assert result["matched_rules"] == []
    assert result["is_secondary"] is False


def test_r9_fires_on_correspondence_subdivision_and_is_not_secondary():
    # A correspondent's own letters are still primary source material, not
    # criticism about them -- fires regardless of whose name it's attached
    # to (here, the recipient of Whitman's letters, not Whitman himself).
    record = _base_record(
        first_author="Whitman, Walt,",
        first_author_dates="1819-1892.",
        subject_genre_fields=[
            _occ("600", [("a", "Doyle, Peter,"), ("x", "Correspondence")])
        ],
    )
    result = sc.classify_record(record)
    assert "R9" in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["confidence"] == "high"
    assert result["primary_document_present"] is True
    assert result["review_flag"] is True


def test_r9_fires_on_interviews_subdivision():
    record = _base_record(
        subject_genre_fields=[
            _occ("600", [("a", "Williams, Tennessee,"), ("x", "Interviews")])
        ]
    )
    result = sc.classify_record(record)
    assert "R9" in result["matched_rules"]
    assert result["is_secondary"] is False


def test_primary_document_present_suppresses_r3_in_literature():
    # "Conversations with Tennessee Williams" case: an interviews collection
    # whose subject headings also include an unrelated topical "In
    # literature" facet. primary_document_present must suppress R3 so the
    # topical facet doesn't override what the Interviews subdivision already
    # establishes about the resource's form.
    record = _base_record(
        subject_genre_fields=[
            _occ("600", [("a", "Williams, Tennessee,"), ("x", "Interviews")]),
            _occ("651", [("a", "Southern States"), ("x", "In literature")]),
        ]
    )
    result = sc.classify_record(record)
    assert "R3" not in result["matched_rules"]
    assert "R9" in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["review_flag"] is True


def test_r3_still_fires_when_primary_document_present_is_false():
    record = _base_record(
        subject_genre_fields=[
            _occ("651", [("a", "Southern States"), ("x", "In literature")])
        ]
    )
    result = sc.classify_record(record)
    assert "R3" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_primary_document_present_suppresses_r1_record_wide():
    # A letters volume that also names a third, unrelated person with no
    # subdivision at all: primary_document_present blocks R1 record-wide,
    # not just for the correspondence occurrence itself.
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ("600", [("a", "Doyle, Peter,"), ("x", "Correspondence")]),
            _occ("600", [("a", "Some Other Person,")]),
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert "R9" in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["review_flag"] is True


def test_self_subject_present_suppresses_r1_for_a_different_named_person():
    # The Toklas/Stein case: 100 is Stein, a bare self-referential 600
    # confirms self_subject_present record-wide, and a *second* 600 names
    # Stein's real-life partner with a non-form subdivision. Toklas isn't a
    # critic writing about Stein; she's a figure within Stein's own
    # experimental memoir -- self_subject_present must gate R1 for this
    # *other* occurrence too, not just the self-referential one.
    record = _base_record(
        first_author="Stein, Gertrude,",
        first_author_dates="1874-1946.",
        subject_genre_fields=[
            _occ("600", [("a", "Stein, Gertrude,"), ("d", "1874-1946.")]),
            _occ(
                "600",
                [("a", "Toklas, Alice B.,"), ("x", "Friends and associates")],
            ),
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["self_subject"] is True
    assert result["review_flag"] is True


def test_genuine_secondary_dual_biography_not_affected_by_self_subject_gate():
    # A critical dual biography of two literary friends, written by an
    # unrelated third biographer: neither friend is the book's own creator,
    # so self_subject_present is false and R1 fires normally for both.
    record = _base_record(
        first_author="Biographer, Some,",
        first_author_dates="1950-",
        subject_genre_fields=[
            _occ("600", [("a", "Friend, One,")]),
            _occ("600", [("a", "Friend, Two,")]),
        ],
    )
    result = sc.classify_record(record)
    assert "R1" in result["matched_rules"]
    assert result["is_secondary"] is True
    assert result["self_subject"] is False


def test_r5_fires_on_biography_subdivision_when_not_self_subject():
    # Subdivision lives on a 650 (not 600), so R1 (which only inspects 600
    # occurrences) does not also fire -- isolates R5.
    record = _base_record(
        first_author="Smith, John,",
        first_author_dates="1900-1980.",
        subject_genre_fields=[
            _occ("650", [("a", "Authors, American,"), ("x", "Biography")])
        ],
    )
    result = sc.classify_record(record)
    assert result["matched_rules"] == ["R5"]
    assert result["confidence"] == "medium"


def test_r5_does_not_fire_when_self_subject_true():
    record = _base_record(
        first_author="Smith, John,",
        first_author_dates="1900-1980.",
        subject_genre_fields=[
            _occ(
                "600", [("a", "Smith, John,"), ("d", "1900-1980."), ("x", "Biography")]
            )
        ],
    )
    result = sc.classify_record(record)
    assert "R5" not in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["review_flag"] is True  # self_subject always reviews


def test_r5_does_not_fire_on_separate_categorical_heading_when_self_subject_true():
    # Memoir/autobiography: a bare self-referential 600 confirms
    # self_subject_present record-wide, which must also suppress R5 for a
    # *separate* categorical 650 "[Class of persons] -- Biography" heading
    # restating the same fact -- not just for the self 600 occurrence
    # itself. Gating per-occurrence (the pre-fix behavior) let this 650
    # fire R5 even though self_subject_present was true.
    record = _base_record(
        first_author="Smith, John,",
        first_author_dates="1900-1980.",
        subject_genre_fields=[
            _occ("600", [("a", "Smith, John,"), ("d", "1900-1980.")]),
            _occ("650", [("a", "Authors, American,"), ("x", "Biography")]),
        ],
    )
    result = sc.classify_record(record)
    assert "R5" not in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["self_subject"] is True
    assert result["review_flag"] is True


# ---------------------- R8 ----------------------


def test_r8_fires_on_criticism_textual_and_is_not_secondary():
    record = _base_record(
        subject_genre_fields=[
            _occ("650", [("a", "Faulkner, William,"), ("x", "Criticism, textual")])
        ]
    )
    result = sc.classify_record(record)
    assert "R8" in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["confidence"] == "high"
    assert result["review_flag"] is True


def test_r8_does_not_also_trigger_r2_for_the_same_value():
    # "Criticism, textual" contains the substring "criticism" and must not
    # also be caught by the general CRITICISM_TERMS check.
    record = _base_record(
        subject_genre_fields=[
            _occ("650", [("a", "Faulkner, William,"), ("x", "Criticism, textual")])
        ]
    )
    result = sc.classify_record(record)
    assert "R2" not in result["matched_rules"]


def test_r2_still_fires_from_a_separate_value_in_the_same_occurrence_as_r8():
    # A different $x on the same occurrence that IS plain "criticism" should
    # still fire R2 independently of the textual-criticism exclusion.
    record = _base_record(
        subject_genre_fields=[
            _occ(
                "650",
                [
                    ("a", "Faulkner, William,"),
                    ("x", "Criticism, textual"),
                    ("x", "History and criticism"),
                ],
            )
        ]
    )
    result = sc.classify_record(record)
    assert "R8" in result["matched_rules"]
    assert "R2" in result["matched_rules"]
    assert result["is_secondary"] is True


# ---------------------- R6 / R7 (confidence-only) ----------------------


def test_r6_raises_confidence_to_high_when_no_secondary_rule_fired():
    record = _base_record(
        subject_genre_fields=[
            _occ("650", [("a", "Detective fiction"), ("v", "Fiction")])
        ]
    )
    result = sc.classify_record(record)
    assert "R6" in result["matched_rules"]
    assert result["is_secondary"] is False
    assert result["confidence"] == "high"
    assert result["review_flag"] is False


def test_r7_raises_confidence_to_high():
    # 40-char 008 with byte 33 ("literary form") = "p" (one of the specific
    # creative-form codes).
    field_008 = " " * 33 + "p" + " " * 6
    assert len(field_008) == 40
    record = _base_record(field_008=field_008)
    result = sc.classify_record(record)
    assert "R7" in result["matched_rules"]
    assert result["confidence"] == "high"


def test_r7_skipped_on_malformed_008():
    record = _base_record(field_008="too short")
    result = sc.classify_record(record)
    assert "R7" not in result["matched_rules"]
    assert result["confidence"] == "low"
    assert result["review_flag"] is True


def test_no_rules_fired_yields_low_confidence_and_review_flag():
    record = _base_record()
    result = sc.classify_record(record)
    assert result["matched_rules"] == []
    assert result["is_secondary"] is False
    assert result["confidence"] == "low"
    assert result["review_flag"] is True


# ---------------------- self_subject / "different 600" interplay ----------------------


def test_self_subject_alone_is_primary_but_review_flagged():
    # Autobiography / published letters by the author themself.
    record = _base_record(
        first_author="Jewett, Sarah Orne,",
        first_author_dates="1849-1909.",
        subject_genre_fields=[
            _occ("600", [("a", "Jewett, Sarah Orne,"), ("d", "1849-1909.")])
        ],
    )
    result = sc.classify_record(record)
    assert result["is_secondary"] is False
    assert result["self_subject"] is True
    assert result["review_flag"] is True


def test_self_subject_plus_separate_non_self_600_suppresses_r1_record_wide():
    # Superseded by the spec's self_subject_present fix: a record whose
    # creator is also its own bare self-referential subject makes any other
    # bare-named 600 more likely to be a figure inside that self-narrative
    # (companion, correspondent, family member) than an independent critic's
    # subject -- so self_subject_present suppresses R1 for the Howells
    # occurrence too, not just Jewett's. review_flag stays on precisely
    # because the residual case (a genuinely independent secondary entry
    # co-occurring with a self-reference) can't be told apart from this one
    # by heading text alone.
    record = _base_record(
        first_author="Jewett, Sarah Orne,",
        first_author_dates="1849-1909.",
        subject_genre_fields=[
            _occ("600", [("a", "Jewett, Sarah Orne,"), ("d", "1849-1909.")]),
            _occ("600", [("a", "Howells, William Dean,"), ("d", "1837-1920.")]),
        ],
    )
    result = sc.classify_record(record)
    assert result["is_secondary"] is False
    assert "R1" not in result["matched_rules"]
    assert result["self_subject"] is True
    assert result["review_flag"] is True


def test_translated_work_700_does_not_trigger_r1():
    # A 700 translator added-entry is never extracted into
    # subject_genre_fields (not in SUBJECT_GENRE_TAGS), so with zero 600
    # occurrences R1 cannot fire regardless of 700 content.
    rec = make_record(
        [
            ("100", "1", [("a", "Author, Some,")]),
            ("700", "1", [("a", "Translator, Some,")]),
        ]
    )
    data = dc.process_record(rec, dc.parse_range_spec("PS"))
    assert data["subject_genre_fields"] == []
    result = sc.classify_record(data)
    assert "R1" not in result["matched_rules"]
    assert result["is_secondary"] is False


def test_multi_author_critical_survey_650_history_and_criticism_no_600():
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ("650", [("a", "American fiction"), ("x", "History and criticism")])
        ],
    )
    result = sc.classify_record(record)
    assert "R1" not in result["matched_rules"]
    assert "R2" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_festschrift_no_100_with_only_600_subjects():
    record = _base_record(
        first_author=None,
        first_author_dates=None,
        subject_genre_fields=[
            _occ("600", [("a", "Faulkner, William,"), ("d", "1897-1962.")])
        ],
    )
    result = sc.classify_record(record)
    assert "R1" in result["matched_rules"]
    assert result["is_secondary"] is True


def test_matched_rules_records_every_fired_rule_id():
    record = _base_record(
        first_author="Smith, John,",
        first_author_dates="1900-1980.",
        subject_genre_fields=[
            _occ(
                "650",
                [
                    ("a", "American fiction"),
                    ("x", "History and criticism"),
                    ("x", "Biography"),
                ],
            )
        ],
    )
    result = sc.classify_record(record)
    assert set(result["matched_rules"]) == {"R2", "R5"}
    assert result["confidence"] == "high"  # max(R2=high, R5=medium)


# ---------------------- classify_pickles / CSV ----------------------


def test_classify_pickles_end_to_end(tmp_path, marc_gz_writer):
    records = [
        [
            ("050", "0", [("a", "PS3553")]),
            ("100", "1", [("a", "Doe, Jane,")]),
            ("245", "0", [("a", "A novel /"), ("b", "a tale of two cities /")]),
            ("264", "1", [("a", "New York :"), ("c", "1990")]),
        ],
        [
            ("050", "0", [("a", "PS3553")]),
            ("650", "0", [("a", "American fiction"), ("x", "History and criticism")]),
            ("245", "0", [("a", "A study /")]),
            ("264", "1", [("a", "Boston :"), ("c", "1990")]),
        ],
        [("050", "0", [("a", "PR6019")])],  # out of scope
    ]
    xml_path = marc_gz_writer(tmp_path / "sample.xml.gz", records)
    all_data = [
        dc.process_record(rec, dc.parse_range_spec("PS"))
        for rec in dc.parse_records(xml_path)
    ]
    import pickle

    pkl_path = tmp_path / "sample.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_data, f)

    df = sc.classify_pickles(str(tmp_path))
    assert len(df) == 2
    assert set(df.columns) == {
        "lccn",
        "title",
        "subtitle",
        "first_author",
        "first_author_dates",
        "year_min",
        "places",
        "publishers",
        "classifications",
        "is_secondary",
        "confidence",
        "matched_rules",
        "review_flag",
        "self_subject",
        "primary_document_present",
        "subject_headings",
    }
    assert df["is_secondary"].tolist() == [False, True]
    assert df.loc[df["is_secondary"], "matched_rules"].iloc[0] == "R2"
    assert df.loc[~df["is_secondary"], "title"].iloc[0] == "A novel /"
    assert df.loc[~df["is_secondary"], "subtitle"].iloc[0] == "a tale of two cities /"
    assert df.loc[~df["is_secondary"], "first_author"].iloc[0] == "Doe, Jane,"
    assert df.loc[~df["is_secondary"], "year_min"].iloc[0] == 1990
    assert (
        df.loc[df["is_secondary"], "subject_headings"].iloc[0]
        == "650: American fiction -- History and criticism"
    )


def test_classify_pickles_raises_on_stale_pickles_missing_new_fields(tmp_path):
    import pickle

    stale_record = {
        "lccn": "1",
        "classifications": ["PS3553"],
        "matches_class_range": True,
        "title": "Old pickle /",
        "year": ["1990"],
        "places": ["New York :"],
        "publishers": ["Pub,"],
        "first_author": "Doe, Jane,",
    }
    with open(tmp_path / "stale.pkl", "wb") as f:
        pickle.dump([stale_record], f)

    try:
        sc.classify_pickles(str(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "data_collection" in str(e)


def test_matched_rules_csv_round_trip(tmp_path):
    df = pd.DataFrame([{"lccn": "1", "matched_rules": ";".join(["R1", "R2"])}])
    csv_path = tmp_path / "out.csv"
    df.to_csv(csv_path, index=False)
    read_back = pd.read_csv(csv_path)
    assert read_back["matched_rules"].iloc[0].split(";") == ["R1", "R2"]
