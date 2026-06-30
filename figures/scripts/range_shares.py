"""Per-range NYC-imprint share series, shared by the range figures (fig5, fig7).

Applies the same "smooth the counts, then take the proportion" rule as
``fig1.compute_city_share`` / ``cross_range.share_series``, but split by the
numeric PS sub-ranges from :mod:`imprints.ps_ranges`. Unlike ``fig1``, records
with **no place of publication** are dropped so each range's share compares
like-with-like (NYC / (NYC + Other)), matching the ``cross_range`` convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from imprints.ps_ranges import RANGE_ORDER, add_range_column

#: ``city_group`` value for records lacking a place of publication.
NO_PLACE = "No place of publication"

#: Default label for the New-York bucket in ``city_group``.
CITY = "New York City"


def counts_matrices(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
    city: str = CITY,
    ranges: list[str] = RANGE_ORDER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(nyc, other)`` wide matrices of placed-record counts.

    Both frames are indexed by every year in ``[start_year, end_year]`` and have
    one column per range in ``ranges`` (missing cells filled with 0). "No place"
    rows and rows outside every range are dropped first.
    """
    df = add_range_column(df)
    df = df[df["range"].notna()]
    df = df[df["city_group"] != NO_PLACE]
    df = df[df["year_min"].between(start_year, end_year)].copy()
    df["year_min"] = df["year_min"].astype(int)
    df["is_city"] = (df["city_group"] == city).astype(int)

    grouped = df.groupby(["year_min", "range"])["is_city"].agg(nyc="sum", n="count")
    grouped["other"] = grouped["n"] - grouped["nyc"]

    years = pd.Index(range(start_year, end_year + 1), name="year_min")
    nyc = (
        grouped["nyc"].unstack("range").reindex(index=years, columns=ranges).fillna(0.0)
    )
    other = (
        grouped["other"]
        .unstack("range")
        .reindex(index=years, columns=ranges)
        .fillna(0.0)
    )
    return nyc, other


def share_matrix(
    nyc: pd.DataFrame,
    other: pd.DataFrame,
    window: int = 5,
    smooth: bool = True,
    min_n: int = 0,
) -> pd.DataFrame:
    """Return per-range NYC share (%) from count matrices.

    Smoothing (if on) is applied to the raw counts before the proportion, so the
    rule matches ``fig1``/``cross_range``. Year/range cells whose (smoothed)
    placed-record total is below ``min_n`` become ``NaN`` so sparse early years
    don't spike a range's line to 0%/100% off one or two records.
    """
    n = nyc.copy()
    o = other.copy()
    if smooth:
        window = max(1, window)
        n = n.rolling(window=window, min_periods=1).mean()
        o = o.rolling(window=window, min_periods=1).mean()
    total = n + o
    share = (n / total.replace(0, np.nan)) * 100
    return share.where(total >= min_n)


def despread_labels(positions: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions apart so none sit within ``min_gap`` of another.

    Returns adjusted positions in the original order; spreading is done top-down
    on a sorted copy so direct-labelled lines/bands stay readable.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i], reverse=True)
    adjusted = dict.fromkeys(order, 0.0)
    prev = None
    for i in order:
        y = positions[i]
        if prev is not None and prev - y < min_gap:
            y = prev - min_gap
        adjusted[i] = y
        prev = y
    return [adjusted[i] for i in range(len(positions))]
