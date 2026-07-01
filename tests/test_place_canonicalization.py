"""Tests for imprints.place_canonicalization."""

from imprints import place_canonicalization as pc


def test_state_variants_collapse_to_same_canonical_form():
    assert pc.canonicalize_place("boston") == "boston"
    assert pc.canonicalize_place("boston ma") == "boston ma"
    assert pc.canonicalize_place("boston mass") == "boston ma"
    assert pc.canonicalize_place("boston massachusetts") == "boston ma"


def test_postal_code_and_period_abbreviation_forms_match():
    assert pc.canonicalize_place("aberdeen sd") == "aberdeen sd"
    assert pc.canonicalize_place("aberdeen s d") == "aberdeen sd"


def test_bare_state_only_entry_returns_postal_code():
    assert pc.canonicalize_place("mass") == "ma"
    assert pc.canonicalize_place("ny") == "ny"


def test_no_place_markers_return_none():
    assert pc.canonicalize_place("s l") is None
    assert pc.canonicalize_place("n p") is None
    assert pc.canonicalize_place("place of publication not identified") is None


def test_non_us_place_without_state_token_passes_through_unchanged():
    assert pc.canonicalize_place("abidjan riviera") == "abidjan riviera"


def test_address_noise_after_state_token_is_truncated():
    assert pc.canonicalize_place("abilene tex ligustrum dr abilene") == "abilene tx"
    assert pc.canonicalize_place("knoxville tenn p o box knoxville") == "knoxville tn"


def test_last_state_match_wins_for_state_named_cities():
    # "Virginia" and "Delaware" are themselves state names but also real
    # towns in other states; the trailing, more specific state should win.
    assert pc.canonicalize_place("virginia minn") == "virginia mn"


def test_bare_washington_is_not_treated_as_state():
    # Washington Square / Washington Court House, OH are real places, not
    # the state of Washington -- see WA's comment in _RAW_STATE_VARIANTS.
    assert pc.canonicalize_place("washington square") == "washington square"
    assert pc.canonicalize_place("washington court house o") == (
        "washington court house o"
    )


def test_mt_prefix_is_not_treated_as_montana():
    assert pc.canonicalize_place("mt horeb wis") == "mt horeb wi"
    assert pc.canonicalize_place("mt") == "mt"


def test_none_and_empty_input():
    assert pc.canonicalize_place(None) is None
    assert pc.canonicalize_place("") is None
    assert pc.canonicalize_place("   ") is None
