#!/usr/bin/env python3
"""Compare the PS New-York-share curve against every other LC range.

Reads the count tables built by ``imprints.cross_range`` and asks whether PS's
rise-then-fall in NYC imprint share is particular to PS, general to LC, or
shared by a subset of ranges (e.g. American literature / history). Emits a
per-range statistics CSV and four figures.

Run after ``python -m imprints.cross_range``:

    python figures/scripts/cross_range.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import style

REPO = Path(__file__).resolve().parents[2]
DEFAULT_COUNTS = REPO / "data/cross_range"
DEFAULT_OUTDIR = REPO / "figures/outputs"

# Subclasses to surface in the small-multiples grid / labelled comparisons.
# PS first so it anchors the grid; the rest are literature + American-focused
# history ranges most relevant to the research question.
HIGHLIGHT_SUBCLASSES = ["PS", "PR", "PQ", "PT", "PN", "PA", "PE", "PZ", "E", "F"]


def load_counts(counts_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(counts_dir / f"counts_{name}.csv")


def share_series(df: pd.DataFrame, key: str, window: int, smooth: bool) -> pd.Series:
    """Smoothed NYC share (%) per year for one key: nyc / (nyc + other) * 100."""
    sub = df[df["key"] == key].set_index("year_min").sort_index()
    counts = sub[["nyc", "other"]]
    if smooth:
        counts = counts.rolling(window=window, min_periods=1).mean()
    total = counts["nyc"] + counts["other"]
    share = (counts["nyc"] / total.replace(0, np.nan)) * 100
    return share.dropna()


def _ols_slope(years, values):
    """OLS slope (% per year) of values on years; nan if <2 points."""
    if len(years) < 2:
        return np.nan
    X = sm.add_constant(np.asarray(years, dtype=float), has_constant="add")
    model = sm.OLS(np.asarray(values, dtype=float), X).fit()
    return float(model.params[1])


def range_stats(share: pd.Series, n_records: int, ps_share: pd.Series) -> dict:
    """Peak, rise/fall slopes, humpiness, and correlation with PS for one range."""
    years = share.index.to_numpy()
    vals = share.to_numpy()
    peak_year = int(years[vals.argmax()])
    peak_share = float(vals.max())
    start_share = float(vals[0])
    end_share = float(vals[-1])

    pre = share.loc[share.index <= peak_year]
    post = share.loc[share.index >= peak_year]
    aligned = pd.concat([share, ps_share], axis=1, join="inner").dropna()

    return {
        "n_records": int(n_records),
        "start_year": int(years[0]),
        "end_year": int(years[-1]),
        "peak_year": peak_year,
        "peak_share": round(peak_share, 2),
        "start_share": round(start_share, 2),
        "end_share": round(end_share, 2),
        "rise_slope": round(_ols_slope(pre.index, pre.to_numpy()), 4),
        "fall_slope": round(_ols_slope(post.index, post.to_numpy()), 4),
        "humpiness": round(peak_share - max(start_share, end_share), 2),
        "corr_with_PS": (
            round(float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])), 3)
            if len(aligned) >= 3
            else np.nan
        ),
    }


def qualifying_keys(df: pd.DataFrame, min_records: int, min_years: int) -> list:
    """Keys with enough placed records and enough year coverage to be stable."""
    totals = df.groupby("key")[["nyc", "other"]].sum().sum(axis=1)
    spans = df.groupby("key")["year_min"].nunique()
    keep = totals[(totals >= min_records) & (spans >= min_years)].index
    return sorted(keep)


def compute_stats(subclass, letter, window, smooth, min_records, min_years):
    """Build the per-range stats DataFrame across both letter and subclass keys."""
    ps_share = share_series(subclass, "PS", window, smooth)
    rows = []
    for level, df in (("subclass", subclass), ("letter", letter)):
        totals = df.groupby("key")[["nyc", "other"]].sum().sum(axis=1)
        for key in qualifying_keys(df, min_records, min_years):
            share = share_series(df, key, window, smooth)
            if share.empty:
                continue
            stats = range_stats(share, totals[key], ps_share)
            stats["level"] = level
            stats["key"] = key
            rows.append(stats)
    out = pd.DataFrame(rows)
    # Humpiness percentile across all qualifying ranges (both levels).
    out["humpiness_pctile"] = (out["humpiness"].rank(pct=True) * 100).round(1)
    cols = [
        "level",
        "key",
        "n_records",
        "start_year",
        "end_year",
        "peak_year",
        "peak_share",
        "start_share",
        "end_share",
        "rise_slope",
        "fall_slope",
        "humpiness",
        "humpiness_pctile",
        "corr_with_PS",
    ]
    return out[cols].sort_values(["level", "humpiness"], ascending=[True, False])


def _spaghetti(df, keys, highlight, title, ylabel, out_path, window, smooth):
    """Thin gray line per key, one highlighted key drawn bold black."""
    style.apply_style()
    fig, ax = plt.subplots()
    for key in keys:
        if key == highlight:
            continue
        s = share_series(df, key, window, smooth)
        if not s.empty:
            ax.plot(s.index, s.values, color="0.75", linewidth=0.6, zorder=1)
    hl = share_series(df, highlight, window, smooth)
    ax.plot(
        hl.index,
        hl.values,
        color=style.COLOR_NYC,
        linewidth=2.0,
        zorder=3,
        label=highlight,
    )
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    style.save_figure(out_path)
    plt.close(fig)


def _ps_vs_rest(special, out_path, window, smooth):
    """PS vs aggregate not-PS, with 95% binomial CIs (as in fig1)."""
    style.apply_style()
    fig, ax = plt.subplots()
    z = 1.96
    for key, color in (("PS", style.COLOR_NYC), ("not_PS", style.COLOR_OTHER)):
        sub = special[special["key"] == key].set_index("year_min").sort_index()
        counts = sub[["nyc", "other"]]
        if smooth:
            counts = counts.rolling(window=window, min_periods=1).mean()
        total = counts["nyc"] + counts["other"]
        p = counts["nyc"] / total.replace(0, np.nan)
        # CIs from raw (unsmoothed) totals so n reflects real sample size.
        raw = sub[["nyc", "other"]]
        raw_total = (raw["nyc"] + raw["other"]).replace(0, np.nan)
        se = (p * (1 - p) / raw_total) ** 0.5
        years = p.index
        ax.fill_between(
            years, (p - z * se) * 100, (p + z * se) * 100, color=color, alpha=0.15
        )
        label = "PS" if key == "PS" else "Rest of dataset (not PS)"
        ax.plot(years, p * 100, color=color, linewidth=1.8, label=label)
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("% imprints published in New York City")
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    style.save_figure(out_path)
    plt.close(fig)


def _small_multiples(subclass, letter, keys, out_path, window, smooth):
    """Grid of mini-panels; each range's curve with PS faint behind as reference."""
    style.apply_style()
    ncols = 3
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.0 * nrows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()
    ps_share = share_series(subclass, "PS", window, smooth)
    for ax, (level, key) in zip(axes, keys):
        df = subclass if level == "subclass" else letter
        ax.plot(ps_share.index, ps_share.values, color="0.8", linewidth=1.0, zorder=1)
        s = share_series(df, key, window, smooth)
        ax.plot(s.index, s.values, color=style.COLOR_NYC, linewidth=1.2, zorder=2)
        ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=0.6)
        ax.set_title(key, fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axes[len(keys) :]:
        ax.set_visible(False)
    fig.supxlabel("Year")
    fig.supylabel("% imprints in NYC (PS faint behind)")
    plt.tight_layout()
    style.save_figure(out_path)
    plt.close(fig)


def _crossing_50(df, keys, out_path, window, smooth, threshold=50.0):
    """Plot only subclasses whose NYC share ever reaches `threshold`.

    Every range is drawn in black, so each is identified by a unique
    linestyle x marker combination (the marker shape is the primary cue, the
    linestyle the secondary one); legend is ordered by peak share.
    """
    series = {}
    for key in keys:
        s = share_series(df, key, window, smooth)
        if not s.empty and s.max() >= threshold:
            series[key] = s
    order = sorted(series, key=lambda k: series[k].max(), reverse=True)

    style.apply_style()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    step = 15  # years between markers
    for i, key in enumerate(order):
        s = series[key]
        ax.plot(
            s.index,
            s.values,
            linewidth=0.9,
            markersize=4,
            markevery=(i % step, step),
            label=key,
            **style.series_style(i),
        )
    ax.axhline(threshold, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("% imprints in NYC")
    ax.set_title(f"LC subclasses that ever reach {threshold:.0f}% NYC imprints")
    ax.legend(ncol=2, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout()
    style.save_figure(out_path)
    plt.close(fig)


def _ps_pz(df, out_path, window, smooth):
    """PS (US literature) vs PZ (fiction & juvenile belles lettres) NYC share."""
    style.apply_style()
    fig, ax = plt.subplots()
    for i, key in enumerate(("PS", "PZ")):
        s = share_series(df, key, window, smooth)
        ax.plot(
            s.index,
            s.values,
            linewidth=1.4,
            markersize=4,
            markevery=(0, 12),
            label=key,
            **style.series_style(i),
        )
    ax.axhline(50, color=style.COLOR_REFERENCE, linestyle="dotted", linewidth=1)
    ax.set_xlabel("Year")
    ax.set_ylabel("% imprints in NYC")
    ax.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    style.save_figure(out_path)
    plt.close(fig)


def print_summary(stats: pd.DataFrame):
    """Focused PS-vs-rest narrative and the ranges most like PS."""
    ps = stats[(stats.level == "subclass") & (stats.key == "PS")].iloc[0]
    print("\n=== PS vs. rest of dataset ===")
    print(
        f"PS: peak {ps.peak_year} at {ps.peak_share}%, "
        f"rise {ps.rise_slope:+.3f}%/yr, fall {ps.fall_slope:+.3f}%/yr, "
        f"humpiness {ps.humpiness} (pctile {ps.humpiness_pctile})"
    )
    print("\nRanges most correlated with PS (subclass level):")
    top = (
        stats[stats.level == "subclass"]
        .dropna(subset=["corr_with_PS"])
        .sort_values("corr_with_PS", ascending=False)
        .head(12)
    )
    for _, r in top.iterrows():
        print(
            f"  {r.key:<4} corr={r.corr_with_PS:+.3f} peak={r.peak_year} "
            f"hump={r.humpiness:>6} n={r.n_records:,}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-dir", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--smooth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-records", type=int, default=2000)
    parser.add_argument("--min-years", type=int, default=40)
    args = parser.parse_args()

    subclass = load_counts(args.counts_dir, "subclass")
    letter = load_counts(args.counts_dir, "letter")
    special = load_counts(args.counts_dir, "special")

    stats = compute_stats(
        subclass, letter, args.window, args.smooth, args.min_records, args.min_years
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_path = args.output_dir / "cross_range_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path} ({len(stats)} ranges)")
    print_summary(stats)

    sub_keys = qualifying_keys(subclass, args.min_records, args.min_years)
    letter_keys = qualifying_keys(letter, args.min_records, args.min_years)

    _spaghetti(
        subclass,
        sub_keys,
        "PS",
        "NYC imprint share by LC subclass",
        "% imprints in NYC",
        args.output_dir / "cross_range_subclass.png",
        args.window,
        args.smooth,
    )
    _spaghetti(
        letter,
        letter_keys,
        "P",
        "NYC imprint share by top-level LC class",
        "% imprints in NYC",
        args.output_dir / "cross_range_top.png",
        args.window,
        args.smooth,
    )
    _ps_vs_rest(special, args.output_dir / "ps_vs_rest.png", args.window, args.smooth)

    # Small multiples: the research-relevant ranges that qualify, PS first.
    # A single-letter prefix (E, F) is identical at both levels -> keep it once.
    panel_keys, seen = [], set()
    for level, available in (("subclass", sub_keys), ("letter", letter_keys)):
        for k in HIGHLIGHT_SUBCLASSES:
            if k in available and k not in seen:
                panel_keys.append((level, k))
                seen.add(k)
    _small_multiples(
        subclass,
        letter,
        panel_keys,
        args.output_dir / "cross_range_small_multiples.png",
        args.window,
        args.smooth,
    )

    _crossing_50(
        subclass,
        sub_keys,
        args.output_dir / "cross_range_crossing50.png",
        args.window,
        args.smooth,
    )

    _ps_pz(
        subclass,
        args.output_dir / "cross_range_ps_pz.png",
        args.window,
        args.smooth,
    )


if __name__ == "__main__":
    main()
