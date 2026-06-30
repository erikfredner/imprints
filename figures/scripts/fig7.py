#!/usr/bin/env python3
"""
Ask how much of the post-peak decline in PS's New-York-imprint share is a
*composition* effect (the mix of numerical sub-ranges shifting toward
lower-NYC ranges) versus a *within-range* effect (ranges themselves publishing
less in New York).

The aggregate NYC share is S_t = sum_r w_{r,t} * s_{r,t}, where s_{r,t} is range
r's NYC share in year t and w_{r,t} its share of placed records that year. Three
outputs:

1. A shift-share decomposition of the change between the peak year and a later
   year into within / between(composition) / interaction terms, with per-range
   composition contributions  ->  figures/outputs/fig7_decomposition.csv
2. A counterfactual time-series figure: actual S_t vs. composition-frozen and
   within-frozen trajectories  ->  figures/outputs/fig7.{png,svg,pdf}
3. A confirming OLS: the post-peak time trend in S_t before and after
   controlling for the author ranges' yearly weights (printed + in the CSV).
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import style
from imprints.ps_ranges import RANGE_LABELS, RANGE_ORDER
from range_shares import counts_matrices

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs/fig7.png"
YEAR_START = 1900
YEAR_END = 2010


def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)


def build_panels(
    df: pd.DataFrame, window: int, smooth: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return ``(shares, weights, totals, aggregate)`` year x range panels.

    Counts are optionally rolling-smoothed before shares/weights are taken, so
    the rule matches fig1/cross_range. ``totals`` are placed records per
    range-year (used to weight the confirming regression); ``aggregate`` is the
    overall NYC share S_t. Shares/aggregate are percentages.
    """
    nyc, other = counts_matrices(df, YEAR_START, YEAR_END, ranges=RANGE_ORDER)
    if smooth:
        window = max(1, window)
        nyc = nyc.rolling(window=window, min_periods=1).mean()
        other = other.rolling(window=window, min_periods=1).mean()

    n = nyc + other
    shares = (nyc / n.replace(0, np.nan)) * 100
    weights = n.div(n.sum(axis=1).replace(0, np.nan), axis=0)
    aggregate = (nyc.sum(axis=1) / n.sum(axis=1).replace(0, np.nan)) * 100
    return shares, weights, n, aggregate


def baseline_shares(shares: pd.DataFrame, t0: int) -> pd.Series:
    """Frozen within-range share for each range.

    Uses the range's share at ``t0`` where it exists, otherwise its first
    observed (post-entry) share. This gives ranges that did not yet exist at the
    base year a defined baseline so the decomposition identity stays exact and
    the counterfactual is computable for every year.
    """
    sbar = shares.loc[t0].copy()
    for key in shares.columns:
        if pd.isna(sbar[key]):
            valid = shares[key].dropna()
            sbar[key] = valid.iloc[0] if not valid.empty else np.nan
    return sbar


def decompose(shares: pd.DataFrame, weights: pd.DataFrame, t0: int, t1: int) -> dict:
    """Shift-share decomposition of S_{t1} - S_{t0} into the three terms.

    Ranges absent at ``t0`` (weight 0) get a baseline share so the identity
    within + between + interaction = delta holds exactly; their compositional
    effect lands in the between term.
    """
    w0 = weights.loc[t0].fillna(0.0)
    w1 = weights.loc[t1].fillna(0.0)
    sbar = baseline_shares(shares, t0)
    s0 = shares.loc[t0].fillna(sbar)
    s1 = shares.loc[t1].fillna(sbar)

    within = w0 * (s1 - s0)
    between = (w1 - w0) * s0
    interaction = (w1 - w0) * (s1 - s0)

    s_t0 = float((w0 * s0).sum())
    s_t1 = float((w1 * s1).sum())
    return {
        "t0": t0,
        "t1": t1,
        "S_t0": s_t0,
        "S_t1": s_t1,
        "delta": s_t1 - s_t0,
        "within": float(within.sum()),
        "between": float(between.sum()),
        "interaction": float(interaction.sum()),
        "between_by_range": between,
    }


def counterfactuals(
    shares: pd.DataFrame, weights: pd.DataFrame, t0: int
) -> pd.DataFrame:
    """Actual, composition-frozen, and within-frozen NYC-share trajectories."""
    w0 = weights.loc[t0].fillna(0.0)
    sbar = baseline_shares(shares, t0)

    actual = (weights * shares).sum(axis=1, min_count=1)
    # Composition frozen at t0: only t0-present ranges (w0>0) contribute.
    within_only = shares.mul(w0, axis=1).sum(axis=1, min_count=1)
    # Within-range shares frozen at baseline; only the mix moves.
    composition_only = weights.mul(sbar, axis=1).sum(axis=1, min_count=1)
    return pd.DataFrame(
        {
            "actual": actual,
            "within_only": within_only,
            "composition_only": composition_only,
        }
    )


def run_ols(
    aggregate: pd.Series,
    shares: pd.DataFrame,
    totals: pd.DataFrame,
    t0: int,
    t1: int,
) -> dict:
    """Confirm the decomposition with two post-peak OLS fits.

    ``aggregate_slope`` is the raw time trend in the overall NYC share. The
    within-range trend regresses each range-year share on year with range fixed
    effects, weighted by placed records — the average within-range slope, which
    isolates the within component without the multicollinearity of stuffing the
    (year-collinear) range weights into one aggregate regression. If the
    within-range slope is close to the aggregate slope, the decline is
    within-range.
    """
    years = [y for y in aggregate.index if t0 <= y <= t1]

    agg = pd.DataFrame({"year": years, "S": aggregate.loc[years].to_numpy()}).dropna()
    base = sm.OLS(agg["S"], sm.add_constant(agg[["year"]])).fit()

    records = []
    for key in RANGE_ORDER:
        for year in years:
            s = shares.loc[year, key]
            n = totals.loc[year, key]
            if pd.notna(s) and n > 0:
                records.append((year, key, float(s), float(n)))
    panel = pd.DataFrame(records, columns=["year", "range", "s", "n"])
    dummies = pd.get_dummies(panel["range"], prefix="r", drop_first=True, dtype=float)
    design = sm.add_constant(pd.concat([panel[["year"]], dummies], axis=1))
    within = sm.WLS(panel["s"], design, weights=panel["n"]).fit()

    span = t1 - t0
    return {
        "aggregate_slope": base.params["year"],
        "aggregate_r2": base.rsquared,
        "within_slope": within.params["year"],
        "within_r2": within.rsquared,
        "aggregate_change": base.params["year"] * span,
        "within_change": within.params["year"] * span,
    }


def write_csv(path: Path, dec: dict, ols: dict) -> None:
    """Write the decomposition, per-range between terms, and OLS to a tidy CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    delta = dec["delta"]
    rows = [
        ("summary", "base_year_t0", dec["t0"]),
        ("summary", "end_year_t1", dec["t1"]),
        ("summary", "S_t0_pct", round(dec["S_t0"], 4)),
        ("summary", "S_t1_pct", round(dec["S_t1"], 4)),
        ("summary", "delta_pct", round(delta, 4)),
        ("component", "within", round(dec["within"], 4)),
        ("component", "between_composition", round(dec["between"], 4)),
        ("component", "interaction", round(dec["interaction"], 4)),
    ]
    if delta:
        rows += [
            ("component_share_of_delta", "within", round(dec["within"] / delta, 4)),
            (
                "component_share_of_delta",
                "between_composition",
                round(dec["between"] / delta, 4),
            ),
            (
                "component_share_of_delta",
                "interaction",
                round(dec["interaction"] / delta, 4),
            ),
        ]
    for key in RANGE_ORDER:
        rows.append(
            ("between_by_range", key, round(float(dec["between_by_range"][key]), 4))
        )
    rows += [
        ("ols", "aggregate_slope_pp_per_yr", round(ols["aggregate_slope"], 5)),
        ("ols", "within_range_slope_pp_per_yr", round(ols["within_slope"], 5)),
        ("ols", "aggregate_change_pp", round(ols["aggregate_change"], 4)),
        ("ols", "within_range_change_pp", round(ols["within_change"], 4)),
        ("ols", "aggregate_r2", round(ols["aggregate_r2"], 4)),
        ("ols", "within_range_r2", round(ols["within_r2"], 4)),
    ]

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["section", "key", "value"])
        writer.writerows(rows)
    print(f"Saved decomposition to: {path}")


def plot_counterfactuals(cf: pd.DataFrame, t0: int, output: Path) -> None:
    """Plot actual vs. composition-frozen vs. within-frozen NYC share."""
    style.apply_style()
    fig, ax = plt.subplots()
    years = cf.index.to_numpy()
    ax.plot(
        years,
        cf["actual"],
        label="Actual",
        markevery=10,
        markersize=4,
        **style.series_style(0),
    )
    ax.plot(
        years,
        cf["within_only"],
        label=f"Composition frozen at {t0}",
        markevery=10,
        markersize=4,
        **style.series_style(1),
    )
    ax.plot(
        years,
        cf["composition_only"],
        label=f"Within-range shares frozen at {t0}",
        markevery=10,
        markersize=4,
        **style.series_style(2),
    )
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)
    ax.axvline(t0, color=style.COLOR_REFERENCE, linestyle="-", linewidth=0.8)
    ax.set_xlim(YEAR_START, YEAR_END)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of placed PS works published in New York City")
    style.percent_yaxis(ax)
    ax.legend()
    plt.tight_layout()
    style.save_figure(output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Decompose the PS NYC-share decline into within vs. composition effects"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--base-year",
        type=int,
        default=None,
        help="Base year t0 (default: the empirical peak of aggregate NYC share)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2000,
        help="Comparison year t1 for the decomposition (default: 2000)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Rolling window size in years for smoothing (default: 5)",
    )
    parser.add_argument(
        "--smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Smooth annual counts before shares/weights (default: true)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: figures/outputs/fig7.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    shares, weights, totals, aggregate = build_panels(df, args.window, args.smooth)

    t0 = args.base_year if args.base_year is not None else int(aggregate.idxmax())
    t1 = args.end_year
    for label, year in (("--base-year", t0), ("--end-year", t1)):
        if year not in aggregate.index:
            raise SystemExit(
                f"{label} {year} is outside the analysis window "
                f"[{YEAR_START}, {YEAR_END}]."
            )
    print(f"Base year t0 (peak): {t0} (S={aggregate.loc[t0]:.2f}%)")
    print(f"End year t1: {t1} (S={aggregate.loc[t1]:.2f}%)")

    dec = decompose(shares, weights, t0, t1)
    recombined = dec["within"] + dec["between"] + dec["interaction"]
    print(
        f"Delta S = {dec['delta']:.2f} pp  |  "
        f"within={dec['within']:.2f}, between(composition)={dec['between']:.2f}, "
        f"interaction={dec['interaction']:.2f}  (sum={recombined:.2f})"
    )
    if dec["delta"]:
        print(
            f"Composition explains {dec['between'] / dec['delta'] * 100:.1f}% of the change; "
            f"within-range {dec['within'] / dec['delta'] * 100:.1f}%."
        )
    print("Largest composition (between) contributions by range:")
    contrib = dec["between_by_range"].sort_values()
    for key, val in contrib.items():
        print(f"  {key} ({RANGE_LABELS[key]}): {val:+.2f} pp")

    ols = run_ols(aggregate, shares, totals, t0, t1)
    print(
        f"OLS {t0}-{t1}: aggregate trend {ols['aggregate_slope']:.4f} pp/yr "
        f"({ols['aggregate_change']:+.1f} pp), within-range trend (range FE, "
        f"records-weighted) {ols['within_slope']:.4f} pp/yr "
        f"({ols['within_change']:+.1f} pp)"
    )

    write_csv(args.output.with_name("fig7_decomposition.csv"), dec, ols)

    cf = counterfactuals(shares, weights, t0)
    plot_counterfactuals(cf, t0, args.output)


if __name__ == "__main__":
    main()
