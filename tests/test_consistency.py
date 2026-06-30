"""Guards for invariants that span modules / data files."""

from imprints import data_cleaning as cl
from imprints import data_collection as dc


# The class-range logic is intentionally duplicated across the two modules
# (see CLAUDE.md). These guard against the copies drifting apart.

SAMPLES = ["PS", "PS3555.123", "PR9053", "PZ3.J55", "813.49", "PSA123", ""]
RANGES = ["PS", "PR9000-PR9999", "PG"]


def test_parse_class_matches_across_modules():
    for s in SAMPLES:
        assert dc.parse_class(s) == cl.parse_class(s), s


def test_parse_range_spec_matches_across_modules():
    for r in RANGES:
        assert dc.parse_range_spec(r) == cl.parse_range_spec(r), r


def test_matches_range_matches_across_modules():
    for prefix in ("P", "PS"):
        for s in SAMPLES:
            assert dc.matches_range(s, prefix) == cl.matches_range(s, prefix), (
                s,
                prefix,
            )


def test_nyc_variants_are_clean_string_idempotent():
    # Each entry in nyc_variants.txt must already be in clean_string() form, or
    # it can never compare equal to a CSV value (which is cleaned). This is the
    # CSV-vs-variants equality invariant from CLAUDE.md.
    with open(cl.NYC_VARIANTS_PATH) as f:
        raw = [line.strip() for line in f if line.strip()]
    assert raw, "nyc_variants.txt should not be empty"
    for entry in raw:
        assert cl.clean_string(entry) == entry, entry
