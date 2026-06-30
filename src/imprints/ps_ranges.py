"""Numeric LC sub-range grouping for the PS class.

The cleaned CSV (``data/PS/data.csv``) carries a ``class_digits`` column: the
integer portion of the first matching LC classification (e.g. ``PS3573.O693`` ->
``3573``). This module turns those integers into the named sub-ranges from the
Library of Congress PS schedule so the figures can compare the NYC-imprint story
*within* PS.

It is the single source of truth for the ``class_digits`` -> range mapping
(the rest of the codebase only ever grouped by alpha prefix, e.g. ``PS`` vs
``PR`` in :mod:`imprints.cross_range`). The bins are mutually exclusive and
cover ~99.86% of PS records; the handful that fall outside every bin (digits in
the 400-490 gap and a few 9990s) map to ``None`` and are meant to be dropped.
"""

from __future__ import annotations

import pandas as pd

#: ``(key, label, lo, hi)`` with inclusive integer bounds on ``class_digits``.
#: Order is the stacking/plotting order (low LC numbers first). Bounds follow
#: the LC PS schedule; the five "individual authors" period ranges are the bulk
#: of PS and are keyed by the author's career era, not the book's print date.
PS_RANGES: list[tuple[str, str, int, int]] = [
    ("PS1-499", "American lit. (general/period/region/genre)", 1, 499),
    ("PS501-689", "Collections", 501, 689),
    ("PS700-893", "Individual authors · Colonial", 700, 893),
    ("PS991-3390", "Individual authors · 19th c.", 991, 3390),
    ("PS3500-3549", "Individual authors · 1900–1960", 3500, 3549),
    ("PS3550-3576", "Individual authors · 1961–2000", 3550, 3576),
    ("PS3600-3626", "Individual authors · 2001–", 3600, 3626),
]

#: Stacking/plotting order of the range keys.
RANGE_ORDER: list[str] = [key for key, _label, _lo, _hi in PS_RANGES]

#: Human-readable label for each range key.
RANGE_LABELS: dict[str, str] = {key: label for key, label, _lo, _hi in PS_RANGES}

#: The largest-by-record-count ranges featured in the Q1 line chart, ordered
#: largest first: Authors 1961-2000, 1900-1960, 2001-, and 19th c.
FEATURED_KEYS: list[str] = [
    "PS3550-3576",
    "PS3500-3549",
    "PS3600-3626",
    "PS991-3390",
]


def assign_range(class_digits) -> str | None:
    """Return the range key whose ``[lo, hi]`` contains ``class_digits``.

    Returns ``None`` for missing values or digits outside every bin.
    """
    if class_digits is None or pd.isna(class_digits):
        return None
    value = int(class_digits)
    for key, _label, lo, hi in PS_RANGES:
        if lo <= value <= hi:
            return key
    return None


def add_range_column(df: pd.DataFrame, column: str = "range") -> pd.DataFrame:
    """Return a copy of ``df`` with a ``column`` of range keys from ``class_digits``."""
    df = df.copy()
    df[column] = df["class_digits"].map(assign_range)
    return df
