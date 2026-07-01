"""Classify PS-range (American literature) MARC records as primary literary
works vs. secondary (criticism/scholarship/reference) works.

Calibrated specifically for LC class PS; do not apply to other P-subclasses
without revalidation -- the genre/national-literature heading conventions
this relies on are American-literature-specific in their $a values, even
though the subdivision logic generalizes.

Consumes the per-record dicts produced by imprints.data_collection
(specifically the `classifications`, `local_call_numbers`, `first_author`,
`first_author_dates`, `subject_genre_fields`, and `field_008` keys added to
process_record() for this purpose) -- it does not re-parse raw MARC.

Known limitations (do not silently absorb -- corroborate with the validation
plan in the project plan before trusting recall numbers):

- FAST-sourced subject headings (2nd indicator "7", $2 "fast") split what an
  LCSH heading would express as one $x subdivision chain into separate facet
  terms. Because R1 is restricted to 2nd-indicator "0" (LCSH) occurrences, a
  record whose *only* 600/610/611/630 occurrences are non-LCSH will never
  trigger R1, even if it's genuinely a bare critical-subject heading. Before
  trusting recall, count the LCSH share vs. other vocabularies in the corpus.
- `self_subject_present` suppresses both R1 (record-wide) and R5 for the
  whole record, not just the self-referential occurrence -- this is what
  fixes memoirs, autobiographies, and self-narratives that also name a close
  associate (e.g. Toklas/Stein), but a record that combines a genuine
  self-reference with an unrelated, genuinely secondary subject entry about
  a *different* person will have that entry's signal suppressed too.
  `review_flag` is set whenever `self_subject_present` is true specifically
  so this surfaces in QA sampling rather than passing silently.
- `primary_document_present` suppresses R1 record-wide and R3 per-occurrence,
  same tradeoff as above, same mitigation (review_flag always set).
- Collective biography/reference works that do not use the literal word
  "criticism" in their subdivision (e.g. "Authors, American $y 20th century
  $x Biography") are caught only via R5 at medium confidence, not R2's high
  confidence -- treat them as a known soft spot, not a hard miss.
- Craft/pedagogy works ("Fiction $x Authorship", "$x Technique") are not
  caught by any rule in v1.
- A possible future QA cross-check against the LC Cutter-range for
  "Biography and criticism" (Table P-PZ40) is NOT implemented here.
- Self-subject name matching (surname + given name + birth year, see
  _name_key) is heuristic: when neither the 100 nor a 600 occurrence carries
  a $d, common names can produce false-positive self-subject matches.
- R8 ("Criticism, textual") always sets review_flag: a minority of records
  using that identical heading are standalone monographs about a work's
  textual/editorial history rather than critical editions of the work
  itself, and that split isn't resolvable from the heading text alone.
- The "fictitious character"/"legendary character" qualifier is matched
  against the full concatenated text of every subfield on the occurrence,
  not just $a/$c and not requiring parentheses -- LC practice puts it in
  $c, sometimes without parens (e.g. "Hawke, Alex -- Fictitious character."),
  so an $a/$c-only or parenthesized-only check would miss real cases.
"""

import os
import pickle
import re

import pandas as pd

from imprints.data_cleaning import get_publishers_year_ints, get_years_ints

PS_SCOPE_RE = re.compile(r"^PS\s*\d")

SUBDIVISION_TAGS = {"600", "610", "611", "630", "650", "651"}
NAME_SUBJECT_TAGS = {"600", "610", "611", "630"}

# Term sets below are matched as case-insensitive substrings against $x/$v/$y
# values on a subject occurrence, after stripping terminal punctuation and
# casefolding (see _normalize_term). CHARACTER_QUALIFIERS is the one
# exception -- matched against the full concatenated subfield text of the
# occurrence, not the subdivisions (see module docstring for why).
TEXTUAL_CRITICISM_TERMS = ("criticism, textual",)
CRITICISM_TERMS = ("criticism",)
IN_LITERATURE_TERMS = ("in literature",)
GENRE_INSTANTIATION_TERMS = (
    "fiction",
    "poetry",
    "drama",
    "juvenile fiction",
    "juvenile literature",
    "literary collections",
)
PRIMARY_DOCUMENT_TERMS = ("correspondence", "interviews")
BIOGRAPHY_TERMS = ("biography", "bio-bibliography")
CHARACTER_QUALIFIERS = ("fictitious character", "legendary character")

# 008 byte 33 codes for specific creative literary forms (poetry, drama,
# novel, short story, letters, humor/satire, speeches, mixed forms, and
# generic fiction "1") -- these corroborate PRIMARY status (R7), despite the
# name; LC restricts code "1" to prose fiction classed in P, so it's treated
# the same as the specific codes rather than as a fallback default.
LITERARY_FORM_PRIMARY_CODES = {"p", "d", "f", "j", "i", "h", "s", "m", "1"}

RULE_CONFIDENCE = {
    "R1": "high",
    "R2": "high",
    "R3": "high",
    "R5": "medium",
}
CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

_REQUIRED_KEYS = (
    "subject_genre_fields",
    "field_008",
    "local_call_numbers",
    "first_author_dates",
)


# ---------------------- Step 1: scope filter ----------------------


def in_ps_scope(record):
    """True iff record falls in LC class PS per 050 (preferred) or 090
    (fallback, local call number).

    Deliberately not data_collection.matches_range/parse_class, which
    require digits immediately adjacent to the prefix (no whitespace
    tolerance) -- raw MARC sometimes has "PS 1234"-style entries that those
    would silently reject.

    "Check 050 first; if absent or empty, check 090" is read as fallback-on-
    ABSENCE, not fallback-on-MISMATCH: if any 050 $a values are present at
    all, that authoritative LC classification is trusted and 090 is not
    consulted, even if none of those 050 values happen to match PS. 090 is a
    fallback for records with no LC classification assigned, not a second
    chance for non-PS-classified records.
    """
    classifications = record.get("classifications") or []
    if classifications:
        return any(bool(c) and PS_SCOPE_RE.match(c.strip()) for c in classifications)
    local_call_numbers = record.get("local_call_numbers") or []
    return any(bool(c) and PS_SCOPE_RE.match(c.strip()) for c in local_call_numbers)


# ---------------------- self_subject name matching ----------------------


def _normalize_name_part(s):
    """Case-fold and strip terminal punctuation."""
    if not s:
        return ""
    return s.strip().rstrip(".,;:").strip().casefold()


def _extract_birth_year(date_str):
    """First 4-digit run in a $d string, treated as a birth year.

    E.g. "1899-1961." -> "1899"; "1899-" -> "1899". This means an author
    with unknown death year ("1899-") still compares equal to a record that
    has the full span ("1899-1961."), per the spec's preference for
    surname+given+birth-year comparison over full date-string equality.
    """
    if not date_str:
        return None
    m = re.search(r"\d{4}", date_str)
    return m.group(0) if m else None


def _name_key(name, dates):
    """(surname, given, birth_year) comparison key, or None if no name.

    Known limitation: when neither side has a $d, the key degrades to
    surname+given only, which can over-match common names.
    """
    cleaned = _normalize_name_part(name)
    if not cleaned:
        return None
    surname, _, given = cleaned.partition(",")
    return (surname.strip(), given.strip(), _extract_birth_year(dates))


def _subfield_text(subfields, code):
    return next((text for c, text in subfields if c == code), None)


def _occ_is_self(occ, creator_key):
    a = _subfield_text(occ["subfields"], "a")
    d = _subfield_text(occ["subfields"], "d")
    occ_key = _name_key(a, d)
    return creator_key is not None and occ_key is not None and occ_key == creator_key


def _self_subject_present(occurrences, creator_key):
    """Record-level fact: true iff any 600 in the record names the record's
    own 100 creator. Computed once from the 600s only and used everywhere
    that references "self" below -- a topical heading (650/651) can never
    textually match a personal name, so re-deriving this per-occurrence for
    topical fields would always read as "not self" even in a record that
    also carries a genuinely self-referential 600."""
    return any(
        _occ_is_self(occ, creator_key) for occ in occurrences if occ["tag"] == "600"
    )


# ---------------------- subdivision substring helpers ----------------------


def _normalize_term(text):
    """Case-fold and strip terminal punctuation from a subfield value, for
    substring matching against the term sets above. Cataloging punctuation
    is inconsistent ("Criticism, interpretation, etc." vs "Criticism and
    interpretation" vs occasional lowercase in non-LC copy)."""
    if not text:
        return ""
    return text.strip().rstrip(".,;:").strip().casefold()


def _occ_subdivision_values(occ):
    """Every $x/$v/$y value on one occurrence, normalized. One 650 can carry
    multiple $x/$y/$v subfields in sequence (a subdivision chain) -- all are
    returned, not just the first."""
    return [
        _normalize_term(text)
        for code, text in occ["subfields"]
        if code in ("x", "v", "y")
    ]


def _occ_full_text_folded(occ):
    """Every subfield value on the occurrence, concatenated and casefolded.
    Used for CHARACTER_QUALIFIERS matching: the qualifier can arrive in $a,
    in a separate $c, with or without parentheses, so this checks the whole
    occurrence's text rather than any specific subfield or punctuation
    form."""
    return " ".join(text for _, text in occ["subfields"] if text).casefold()


def _occ_has_character_qualifier(occ):
    return any(q in _occ_full_text_folded(occ) for q in CHARACTER_QUALIFIERS)


def _all_subdivision_values(occurrences):
    """(occurrence, normalized value) pairs for every $x/$v/$y across all
    600/610/611/630/650/651 occurrences in the record."""
    for occ in occurrences:
        if occ["tag"] not in SUBDIVISION_TAGS:
            continue
        for value in _occ_subdivision_values(occ):
            yield occ, value


def _any_value_matches(occurrences, terms, exclude_terms=()):
    """True if any $x/$v/$y value (record-wide) contains one of `terms` as a
    substring. `exclude_terms` skips values that themselves match a more
    specific term set -- used so a "Criticism, textual" value doesn't also
    count as plain "Criticism" (R8 before R2), without suppressing R2 for a
    *different* value/occurrence that is genuinely just "Criticism"."""
    for _occ, value in _all_subdivision_values(occurrences):
        if exclude_terms and any(t in value for t in exclude_terms):
            continue
        if any(t in value for t in terms):
            return True
    return False


def _rule_r8_textual_criticism(occurrences):
    return _any_value_matches(occurrences, TEXTUAL_CRITICISM_TERMS)


def _rule_r2_criticism(occurrences):
    return _any_value_matches(
        occurrences, CRITICISM_TERMS, exclude_terms=TEXTUAL_CRITICISM_TERMS
    )


def _rule_r3_in_literature(occurrences, primary_document_present):
    """IN_LITERATURE_TERMS, gated off record-wide when
    primary_document_present is true: a topical "In literature" facet on a
    record that's fundamentally a correspondence/interviews collection
    almost always describes a topic touched on *within* those letters/
    interviews, not a claim that the whole resource is criticism about that
    topic (the *Conversations with Tennessee Williams* case)."""
    if primary_document_present:
        return False
    return _any_value_matches(occurrences, IN_LITERATURE_TERMS)


def _rule_r6_genre_instantiation(occurrences):
    return _any_value_matches(occurrences, GENRE_INSTANTIATION_TERMS)


def _rule_r9_primary_document(occurrences):
    """Fires regardless of tag or whose name it's attached to -- a
    correspondent's or interviewer's own letters/interviews are still
    primary source material, not criticism about them."""
    return _any_value_matches(occurrences, PRIMARY_DOCUMENT_TERMS)


def _rule_r5_biography(occurrences, self_subject_present):
    """BIOGRAPHY_TERMS on any occurrence, gated by the record-level
    self_subject_present fact (not a per-occurrence self-match). A record
    combining a bare self-referential 600 (autobiography entry) with a
    separate categorical 650 "[Class of persons] -- Biography" restating the
    same fact must have R5 suppressed for *both* -- gating per-occurrence
    would let the 650 fire R5 even though it's just describing the same
    self-reference the 600 already confirmed."""
    if self_subject_present:
        return False
    return _any_value_matches(occurrences, BIOGRAPHY_TERMS)


def _classify_name_subject_occurrence(
    occ, self_subject_present, primary_document_present
):
    """Return "R1", "ambiguous", or None for one 600/610/611/630 occurrence.

    R1 (bare name-as-subject, e.g. the "Four American Poets" case) fires
    only if this occurrence didn't already match a more specific subdivision
    rule (R8/R2/R3/R6) -- checked here via has_prior_signal, since those are
    otherwise evaluated record-wide above. "ambiguous" is LC's identical
    bare heading used both for a work in which a character *appears* and for
    criticism *about* the character/legend -- unresolvable from the heading
    text alone, so no rule fires but the record is flagged for human review.

    R1 is gated by the record-level self_subject_present and
    primary_document_present facts, not by re-comparing this occurrence's
    own $a to the creator -- that's what correctly suppresses R1 on
    `Toklas, Alice B.` in a Stein self-narrative (self_subject_present is
    true because of a *different* 600), not just on `Stein` itself.

    R1 is also restricted to second-indicator "0" (LCSH) occurrences: FAST
    and other non-LCSH vocabularies frequently duplicate an LCSH-subdivided
    heading (e.g. "610 Templars -- Fiction." LCSH) as a separate bare
    heading with the subdivision stripped (e.g. "610 Templars. $2 fast").
    Without this restriction, that bare FAST duplicate would fire R1 on a
    record whose LCSH sibling heading already correctly matched R6.
    """
    if occ["tag"] not in NAME_SUBJECT_TAGS:
        return None

    values = _occ_subdivision_values(occ)
    has_qualifier = _occ_has_character_qualifier(occ)
    is_lcsh = occ.get("ind2") == "0"
    has_prior_signal = any(
        t in v
        for v in values
        for t in (
            TEXTUAL_CRITICISM_TERMS
            + CRITICISM_TERMS
            + IN_LITERATURE_TERMS
            + GENRE_INSTANTIATION_TERMS
        )
    )

    if (
        is_lcsh
        and not self_subject_present
        and not primary_document_present
        and not has_qualifier
        and not has_prior_signal
    ):
        return "R1"
    if has_qualifier and not values:
        return "ambiguous"
    return None


def _parse_008(field_008):
    """Return (literary_form_code, biography_code) at 0-indexed bytes 33, 34,
    or (None, None) if field_008 is missing/short. Never raises."""
    if not field_008 or len(field_008) != 40:
        return None, None
    return field_008[33], field_008[34]


def _rule_r7_literary_form(field_008):
    literary_form, _biography_code = _parse_008(field_008)
    return literary_form in LITERARY_FORM_PRIMARY_CODES


# ---------------------- Step 3: classification ----------------------


def classify_record(record):
    """Apply rules R1-R3, R5-R9 to a single record dict (as produced by
    imprints.data_collection.process_record) and return:
        {"is_secondary": bool, "confidence": "high"|"medium"|"low",
         "matched_rules": [str, ...], "review_flag": bool,
         "self_subject": bool, "primary_document_present": bool}
    """
    occurrences = record.get("subject_genre_fields") or []
    creator_key = _name_key(
        record.get("first_author"), record.get("first_author_dates")
    )

    self_subject_present = _self_subject_present(occurrences, creator_key)
    primary_document_present = _rule_r9_primary_document(occurrences)

    r1_fired = False
    ambiguous = False
    for occ in occurrences:
        outcome = _classify_name_subject_occurrence(
            occ, self_subject_present, primary_document_present
        )
        if outcome == "R1":
            r1_fired = True
        elif outcome == "ambiguous":
            ambiguous = True

    matched = set()
    if r1_fired:
        matched.add("R1")
    if _rule_r2_criticism(occurrences):
        matched.add("R2")
    if _rule_r3_in_literature(occurrences, primary_document_present):
        matched.add("R3")
    if _rule_r5_biography(occurrences, self_subject_present):
        matched.add("R5")
    if _rule_r6_genre_instantiation(occurrences):
        matched.add("R6")
    if _rule_r7_literary_form(record.get("field_008")):
        matched.add("R7")
    if _rule_r8_textual_criticism(occurrences):
        matched.add("R8")
    if primary_document_present:
        matched.add("R9")

    matched_rules = [
        r for r in ("R1", "R2", "R3", "R5", "R6", "R7", "R8", "R9") if r in matched
    ]

    is_secondary = any(r in matched for r in ("R1", "R2", "R3", "R5"))

    if is_secondary:
        fired_secondary = [r for r in matched_rules if r in RULE_CONFIDENCE]
        confidence = max(
            (RULE_CONFIDENCE[r] for r in fired_secondary), key=CONFIDENCE_RANK.get
        )
    elif matched & {"R6", "R7", "R8", "R9"}:
        confidence = "high"
    else:
        confidence = "low"

    review_flag = (
        self_subject_present
        or primary_document_present
        or ambiguous
        or "R8" in matched
        or (not is_secondary and confidence == "low")
    )

    return {
        "is_secondary": is_secondary,
        "confidence": confidence,
        "matched_rules": matched_rules,
        "review_flag": review_flag,
        "self_subject": self_subject_present,
        "primary_document_present": primary_document_present,
    }


# ---------------------- review-CSV metadata helpers ----------------------


def _render_subject_occurrence(occ):
    """One 600/610/611/630/650/651/655 occurrence as "TAG: $a -- $x -- ...",
    for a human reviewer to see at a glance which heading fired a rule."""
    parts = [text for _, text in occ["subfields"]]
    return f"{occ['tag']}: {' -- '.join(parts)}"


def _render_subject_headings(occurrences):
    return "; ".join(_render_subject_occurrence(o) for o in occurrences)


def _year_min(record):
    """Earliest year found in either the year or publisher fields, mirroring
    data_cleaning.cleaning_pipeline's year_min so it's comparable/joinable
    with data/PS/data.csv by eye."""
    candidates = [
        v
        for v in (
            get_years_ints(record.get("year")),
            get_publishers_year_ints(record.get("publishers")),
        )
        if v is not None
    ]
    return min(candidates) if candidates else None


# ---------------------- pickle loading / CLI ----------------------


def load_records(pickle_dir):
    records = []
    for file_name in sorted(os.listdir(pickle_dir)):
        if file_name.endswith(".pkl"):
            with open(os.path.join(pickle_dir, file_name), "rb") as f:
                records.extend(pickle.load(f))
    return records


def classify_pickles(pickle_dir):
    """Load every PS pickle in `pickle_dir`, restrict to in-scope records,
    classify each, and return a DataFrame keyed by lccn.

    Includes descriptive metadata (title, subtitle, author, year, place,
    publisher, classification, and a rendered subject-heading summary) alongside the
    classification result columns, purely to make manual review (spec Step
    5) legible without a separate join against data/PS/data.csv -- note
    that data.csv is exploded per place-of-publication and so is not
    uniquely keyed by lccn, whereas this CSV is one row per record.

    Raises RuntimeError if the pickles predate the fields this classifier
    needs (i.e. were produced before data_collection.py was extended), since
    silently classifying every record as "no rules fired" would be wrong,
    not just incomplete.
    """
    records = load_records(pickle_dir)
    if records and not all(k in records[0] for k in _REQUIRED_KEYS):
        raise RuntimeError(
            f"Pickles in {pickle_dir!r} are missing fields required by "
            "secondary_classification (subject_genre_fields/field_008/"
            "local_call_numbers/first_author_dates). Regenerate them with: "
            "python -m imprints.data_collection --input_dir data/raw "
            "--output_dir data --class_range PS"
        )

    rows = []
    for record in records:
        if not in_ps_scope(record):
            continue
        result = classify_record(record)
        rows.append(
            {
                "lccn": record.get("lccn"),
                "title": record.get("title"),
                "subtitle": record.get("subtitle"),
                "first_author": record.get("first_author"),
                "first_author_dates": record.get("first_author_dates"),
                "year_min": _year_min(record),
                "places": record.get("places"),
                "publishers": record.get("publishers"),
                "classifications": record.get("classifications"),
                "is_secondary": result["is_secondary"],
                "confidence": result["confidence"],
                "matched_rules": ";".join(result["matched_rules"]),
                "review_flag": result["review_flag"],
                "self_subject": result["self_subject"],
                "primary_document_present": result["primary_document_present"],
                "subject_headings": _render_subject_headings(
                    record.get("subject_genre_fields") or []
                ),
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify PS-range MARC records as primary or secondary literature."
    )
    parser.add_argument(
        "--pickle_dir", default="data/PS", help="Directory with PS .pkl files"
    )
    parser.add_argument(
        "--output_csv",
        default="data/PS/secondary_classification.csv",
        help="CSV file for output",
    )
    args = parser.parse_args()

    df = classify_pickles(args.pickle_dir)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df):,} classified records to {args.output_csv}")
