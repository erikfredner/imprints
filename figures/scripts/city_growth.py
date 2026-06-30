#!/usr/bin/env python3
"""Rank and plot the growth of top non-NYC PS imprint locations (1950-2010).

The essay shows New York City's *share* of PS (US literary) imprints falling
after its ~1958 peak. This script tests whether that is mostly a denominator
effect -- not NYC shrinking, but everywhere else growing -- by ranking and
plotting the absolute growth of the top publishing locations *outside* NYC over
the period of NYC's relative decline.

It reads the cleaned CSV, splits a city from its (inconsistently encoded) state,
folds obvious typos into their canonical spelling via string similarity, then:

* plots the top cities by total record count (raw counts vs. year),
* prints a ranking by absolute growth (late-window minus early-window),
* flags city names that are really several different places (e.g. Portland
  OR vs. ME) that each grow, and
* writes the full ranking to a CSV.

Standalone like the other figure scripts: ``python figures/scripts/city_growth.py``.
"""

import argparse
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import style

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/city_growth.png"

#: Window over which NYC's share declines (1950 sits just before its ~1958 peak,
#: capturing the run-up as well as the decline).
YEAR_START = 1950
YEAR_END = 2010

#: Number of years at each end of the window used to measure absolute growth.
GROWTH_WINDOW = 5

#: ``city_group`` value to keep: everything that is neither NYC nor missing.
OTHER = "Other"

#: Substrings (in cleaned ``publisher_clean``) that mark a large-print / reprint
#: house. These reprint existing titles in large-print editions rather than
#: originating literary publishing, and a handful of them (clustered in
#: Waterville/Thorndike, ME) dominate the raw counts. They are filtered out for
#: the "no large print" variant of the figure. Two kinds of marker: the phrase
#: "large print" (and variants), and the distinctive names of the major houses
#: that don't carry it in their imprint string. Matched by substring, so
#: e.g. "thorndike press" and "center point pub" both hit. Curated like
#: ``nyc_variants.txt`` -- extend here if new reprint houses surface.
LARGE_PRINT_MARKERS = (
    "large print",
    "large type",
    "lg print",
    "thorndike",
    "g k hall",
    "wheeler pub",
    "wheeler publishing",
    "center point",
    "five star",
    "chivers",
    "curley",
    "ulverscroft",
    "isis large",
    "magna print",
)

#: Trailing tokens that mark a country, stripped before state detection so e.g.
#: "new york n y u s a" reduces to "new york" + NY.
COUNTRY_TAILS = frozenset({"u s a", "usa", "u s", "u s of a"})

#: Curated map of trailing region phrase -> canonical 2-letter US state. Place
#: strings encode state every which way: clean USPS codes ("louisville ky"),
#: old-style abbreviations ("calif", "mass", "tex"), space-fragmented forms
#: ("garden city n y" -- clean_string drops the periods in "N.Y."), and full
#: names ("ohio"). Bare foreign/stateless cities ("london") match nothing here
#: and pass through whole. If this grows, it could graduate to a curated text
#: file like nyc_variants.txt.
STATE_TOKENS = {
    # Canonical USPS two-letter codes.
    **{
        c: c.upper()
        for c in (
            "al ak az ar ca co ct de fl ga hi id il in ia ks ky la me md ma mi "
            "mn ms mo mt ne nv nh nj nm ny nc nd oh ok or pa ri sc sd tn tx ut "
            "vt va wa wv wi wy dc"
        ).split()
    },
    # Space-fragmented two-letter abbreviations ("N.Y." -> "n y").
    "n y": "NY",
    "n j": "NJ",
    "n h": "NH",
    "n c": "NC",
    "n m": "NM",
    "n d": "ND",
    "s c": "SC",
    "s d": "SD",
    "r i": "RI",
    "w v": "WV",
    "w va": "WV",
    "d c": "DC",
    # Old-style (AP / pre-USPS) abbreviations.
    "ala": "AL",
    "ariz": "AZ",
    "ark": "AR",
    "calif": "CA",
    "colo": "CO",
    "conn": "CT",
    "del": "DE",
    "fla": "FL",
    "ill": "IL",
    "ind": "IN",
    "kan": "KS",
    "kans": "KS",
    "mass": "MA",
    "mich": "MI",
    "minn": "MN",
    "miss": "MS",
    "mont": "MT",
    "neb": "NE",
    "nebr": "NE",
    "nev": "NV",
    "okla": "OK",
    "ore": "OR",
    "oreg": "OR",
    "penn": "PA",
    "penna": "PA",
    "tenn": "TN",
    "tex": "TX",
    "wash": "WA",
    "wis": "WI",
    "wisc": "WI",
    "wyo": "WY",
    # Full state names.
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "wisconsin": "WI",
    "wyoming": "WY",
}


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV (only the columns this script needs)."""
    return pd.read_csv(
        csv_path,
        usecols=["places_clean", "city_group", "year_min", "publisher_clean"],
    )


def is_large_print(publisher: str) -> bool:
    """True if ``publisher_clean`` names a large-print / reprint house."""
    if not isinstance(publisher, str):
        return False
    return any(marker in publisher for marker in LARGE_PRINT_MARKERS)


def _strip_country_tail(tokens: list[str]) -> list[str]:
    """Drop a trailing country marker (e.g. "u s a") if present."""
    for n in (3, 2, 1):
        if len(tokens) > n and " ".join(tokens[-n:]) in COUNTRY_TAILS:
            return tokens[:-n]
    return tokens


def split_city_state(place: str) -> tuple[str | None, str | None]:
    """Split a cleaned place string into ``(city, state)``.

    Greedily strips a trailing region phrase found in :data:`STATE_TOKENS`
    (longest match first), recording its canonical 2-letter state. Whatever
    remains is the city. Strings with no recognized region keep ``state=None``
    and stay whole, so foreign/stateless places (``london``) survive intact.
    Returns ``(None, None)`` when nothing usable is left.
    """
    if not isinstance(place, str):
        return None, None
    tokens = _strip_country_tail(place.split())
    if not tokens:
        return None, None
    state = None
    # Try the trailing two tokens, then one, so "n y" beats a stray "y".
    for n in (2, 1):
        if len(tokens) > n:
            tail = " ".join(tokens[-n:])
            if tail in STATE_TOKENS:
                state = STATE_TOKENS[tail]
                tokens = tokens[:-n]
                break
    city = " ".join(tokens).strip()
    return (city or None), state


def parse_places(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``city`` and ``state`` columns; drop rows with no usable city."""
    parsed = df["places_clean"].map(split_city_state)
    df = df.assign(
        city=parsed.map(lambda cs: cs[0]),
        state=parsed.map(lambda cs: cs[1]),
    )
    return df[df["city"].notna()].copy()


def unrecognized_trailing_tokens(df: pd.DataFrame, min_count: int = 25) -> Counter:
    """Count trailing tokens that were *not* matched as a state.

    These are the strings to inspect when extending :data:`STATE_TOKENS` -- the
    same transparency loop ``get_unique_places`` provides for ``nyc_variants``.
    A row's trailing token is "unrecognized" only when no state was detected.
    """
    no_state = df[df["state"].isna()]
    tails = no_state["places_clean"].dropna().str.split().str[-1]
    counts = Counter(tails)
    return Counter({t: n for t, n in counts.items() if n >= min_count})


def fuzzy_merge(
    counts: pd.Series, threshold: float
) -> tuple[dict[str, str], list[tuple[str, str, float, int]]]:
    """Fold low-count city spellings into a similar high-count canonical one.

    ``counts`` is per-city total N, descending. Each city, in turn, becomes a
    canonical anchor; any later (lower-count) city sharing its first letter and
    scoring >= ``threshold`` on :class:`difflib.SequenceMatcher` is mapped onto
    it. Anchor-only comparison keeps this tractable and biases merges toward
    typo -> canonical (e.g. ``altanta`` -> ``atlanta``).

    Returns ``(mapping, merges)`` where ``mapping`` sends every city to its
    canonical name and ``merges`` logs ``(variant, canonical, score, count)``.
    """
    cities = list(counts.index)
    mapping = {c: c for c in cities}
    merges: list[tuple[str, str, float, int]] = []
    anchors: list[str] = []
    for city in cities:  # already sorted by count, descending
        best_anchor, best_score = None, 0.0
        for anchor in anchors:
            if anchor[:1] != city[:1]:
                continue
            score = SequenceMatcher(None, city, anchor).ratio()
            if score > best_score:
                best_anchor, best_score = anchor, score
        if best_anchor is not None and best_score >= threshold:
            mapping[city] = best_anchor
            merges.append((city, best_anchor, best_score, int(counts[city])))
        else:
            anchors.append(city)
    return mapping, merges


def annual_counts(df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """Wide matrix of per-year record counts: index = year, columns = city."""
    years = pd.Index(range(start, end + 1), name="year_min")
    return (
        df.groupby(["year_min", "city"])
        .size()
        .unstack("city", fill_value=0)
        .reindex(years, fill_value=0)
    )


def growth_table(counts: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rank cities by absolute growth = late-window minus early-window mean/yr.

    Columns: ``total`` (records over the whole span), ``early`` and ``late``
    (mean records/yr in the first/last ``window`` years), and ``growth``
    (late - early). Sorted by ``growth`` descending.
    """
    early = counts.iloc[:window].mean()
    late = counts.iloc[-window:].mean()
    table = pd.DataFrame(
        {
            "total": counts.sum().astype(int),
            "early": early,
            "late": late,
            "growth": late - early,
        }
    )
    return table.sort_values("growth", ascending=False)


def flag_homonyms(
    df: pd.DataFrame,
    window: int,
    start: int,
    end: int,
    min_state_n: int = 10,
    min_growth: float = 1.0,
) -> list[tuple[str, pd.DataFrame]]:
    """Find city names that are really >=2 *different* places, each growing.

    A merged city line (e.g. "Columbus") can hide distinct towns in different
    states (OH vs. MS). For each name we break records down by detected state
    and qualify a state when it holds >= ``min_state_n`` records and grows by
    >= ``min_growth`` records/yr (late-window minus early-window mean). A name
    is flagged when two or more *named* states qualify -- so the merged line
    really should be split. Records with no detected state (foreign/stateless,
    shown as "??") are reported for context but never qualify, since they are
    usually the same place written without a state, not a separate one.
    """
    flagged: list[tuple[str, pd.DataFrame]] = []
    for city, group in df.groupby("city"):
        total = len(group)
        rows = []
        for state, sgroup in group.groupby(group["state"].fillna("??")):
            series = annual_counts(sgroup.assign(city="x"), start, end).iloc[:, 0]
            growth = series.iloc[-window:].mean() - series.iloc[:window].mean()
            qualifies = (
                state != "??" and len(sgroup) >= min_state_n and growth >= min_growth
            )
            rows.append((state, len(sgroup), len(sgroup) / total, growth, qualifies))
        if sum(r[4] for r in rows) >= 2:
            detail = pd.DataFrame(
                rows, columns=["state", "n", "share", "growth", "qualifies"]
            ).sort_values("n", ascending=False)
            flagged.append((city, detail))
    return flagged


def dominant_publishers(df: pd.DataFrame, cities: list[str]) -> dict[str, str]:
    """Most common ``publisher_clean`` per city, to explain artifact spikes."""
    out = {}
    for city in cities:
        pubs = df.loc[df["city"] == city, "publisher_clean"].dropna()
        out[city] = pubs.mode().iloc[0] if not pubs.empty else "(unknown)"
    return out


def plot(counts: pd.DataFrame, top_cities: list[str], output: Path) -> None:
    """Line chart of raw annual counts for the top cities."""
    style.apply_style()
    plt.figure()
    years = counts.index.to_numpy()
    for i, city in enumerate(top_cities):
        plt.plot(
            years,
            counts[city].to_numpy(),
            label=city.title(),
            markevery=5,
            markersize=4,
            **style.series_style(i),
        )
    plt.xlabel("Year")
    plt.ylabel("PS records")
    plt.legend()
    plt.tight_layout()
    style.save_figure(output)


def emit_variant(df: pd.DataFrame, label: str, output: Path, args) -> None:
    """Rank, plot, and report the top cities for one slice of the data.

    ``df`` is the parsed/merged data (already filtered to a publisher subset for
    the "no large print" variant). Writes the figure and a ranking CSV beside
    ``output`` and prints the plotted top cities plus the growth ranking.
    """
    counts = annual_counts(df, args.start_year, args.end_year)
    ranking = growth_table(counts, GROWTH_WINDOW)
    top_cities = list(
        ranking.sort_values("total", ascending=False).head(args.top).index
    )

    plot(counts, top_cities, output)
    ranking_csv = output.with_name(f"{output.stem}_ranking.csv")
    ranking.to_csv(ranking_csv, index_label="city")
    print(f"Saved ranking to: {ranking_csv}")

    span = f"{args.start_year}-{args.end_year}"
    print(f"\n=== Top {args.top} cities by total records, {span} ({label}) ===")
    pubs = dominant_publishers(df, top_cities)
    for city in top_cities:
        row = ranking.loc[city]
        print(
            f"{city.title():<22} total={int(row['total']):>7,}  "
            f"growth={row['growth']:+7.1f}/yr  top publisher: {pubs[city]}"
        )

    print(f"=== Top 15 by absolute growth ({GROWTH_WINDOW}-yr late - early) ===")
    for city, row in ranking.head(15).iterrows():
        print(
            f"{city.title():<22} growth={row['growth']:+7.1f}/yr  "
            f"(early={row['early']:.1f}, late={row['late']:.1f}, "
            f"total={int(row['total']):,})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank and plot growth of top non-NYC PS imprint locations"
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-year", type=int, default=YEAR_START)
    parser.add_argument("--end-year", type=int, default=YEAR_END)
    parser.add_argument("--top", type=int, default=8, help="Cities to plot")
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.9,
        help="Min difflib ratio to fold a spelling into a canonical city",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    df = df[df["city_group"] == OTHER]
    df = df[df["year_min"].between(args.start_year, args.end_year)].copy()
    df["year_min"] = df["year_min"].astype(int)
    df = parse_places(df)
    df["is_large_print"] = df["publisher_clean"].map(is_large_print)

    # Fold typos into canonical spellings (by descending raw count).
    raw_counts = df["city"].value_counts()
    mapping, merges = fuzzy_merge(raw_counts, args.similarity)
    df["city"] = df["city"].map(mapping)

    # Two figures: all publishers, then again with large-print/reprint houses
    # removed so the artifact (Waterville/Thorndike, ME) doesn't swamp the rest.
    emit_variant(df, "all publishers", args.output, args)
    no_lp_output = args.output.with_name(f"{args.output.stem}_no_large_print.png")
    n_dropped = int(df["is_large_print"].sum())
    print(f"\n[Excluding {n_dropped:,} large-print/reprint records for next variant]")
    emit_variant(
        df[~df["is_large_print"]],
        "no large print",
        no_lp_output,
        args,
    )

    # Full merge log to CSV for audit; print only the higher-impact folds so
    # stdout isn't buried under the long tail of n=1 typos.
    merges.sort(key=lambda m: m[3], reverse=True)
    merges_csv = args.output.with_name(f"{args.output.stem}_merges.csv")
    pd.DataFrame(
        merges, columns=["variant", "canonical", "score", "folded_count"]
    ).to_csv(merges_csv, index=False)
    print(f"\nSaved {len(merges)} fuzzy merges to: {merges_csv}")
    inline_floor = 5
    shown = [m for m in merges if m[3] >= inline_floor]
    print(
        f"=== Fuzzy spelling merges (>= {args.similarity}), "
        f"folds with n >= {inline_floor} ==="
    )
    if shown:
        for variant, canonical, score, count in shown:
            print(f"{variant!r} -> {canonical!r}  score={score:.3f}  n={count:,}")
        print(f"({len(merges) - len(shown)} more folds with n < {inline_floor})")
    else:
        print("(none)")

    flagged = flag_homonyms(df, GROWTH_WINDOW, args.start_year, args.end_year)
    print("\n=== Homonym flags (one name, multiple growing states) ===")
    if flagged:
        for city, detail in flagged:
            print(f"\n{city.title()}:")
            for _, r in detail.iterrows():
                mark = " *" if r["qualifies"] else ""
                print(
                    f"  {r['state']:<4} n={int(r['n']):>6,}  "
                    f"share={r['share']:.0%}  growth={r['growth']:+.1f}/yr{mark}"
                )
        print("\n(* = qualifying state; lines flagged when >= 2 qualify)")
    else:
        print("(none)")

    unrecognized = unrecognized_trailing_tokens(df)
    print("\n=== Top unrecognized trailing tokens (extend STATE_TOKENS?) ===")
    for token, count in unrecognized.most_common(25):
        print(f"{count:>7,}  {token}")


if __name__ == "__main__":
    main()
