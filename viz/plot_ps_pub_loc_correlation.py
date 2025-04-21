#!/usr/bin/env python3
"""
Analyse and visualise the relationship between the annual number of **unique
publishers** and the number of PS‑class American‑literature imprints that list a
place of publication **outside New York City**.

The script now

1.  Plots rolling correlations (unchanged).
2.  **Quantifies the simple correlation** between the two series (r, N, p‑value).
3.  **Tests for autocorrelation** in the bivariate OLS residuals
   ‑ Durbin–Watson
   ‑ Newey–West HAC‑robust coefficient & p.
4.  **Runs Granger‑causality tests** (publisher growth → decentralisation and
   vice‑versa) on first‑differenced, mean‑zero series.
5.  Prints all diagnostics to stdout.

Dependencies
------------
numpy, pandas, matplotlib, scipy, statsmodels (>=0.14), argparse
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests

import warnings

# Silence the FutureWarning raised by statsmodels.grangercausalitytests
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"statsmodels\.tsa\.stattools"
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_data(csv_path: str, start_year: int, end_year: int):
    """Load cleaned PS data and filter by year range."""
    df = pd.read_csv(csv_path)
    if "year_min" not in df.columns:
        raise KeyError("`year_min` column not found in PS data CSV")

    df = df.copy()
    df["year_min"] = pd.to_numeric(df["year_min"], errors="coerce")
    df = df.dropna(subset=["year_min"])
    df["year_min"] = df["year_min"].astype(int)

    return df[(df["year_min"] >= start_year) & (df["year_min"] <= end_year)]


def compute_annual_counts(df: pd.DataFrame, city: str, start_year: int, end_year: int):
    """
    Return a DataFrame indexed by year with columns

    - pub_count   : unique publishers
    - nyc_count   : # imprints in `city`
    - other_count : # imprints outside `city`
    """
    # unique publishers
    pub_counts = df.groupby("year_min")["publisher_clean"].nunique()

    # flag NYC vs other
    df = df.copy()
    df["in_city"] = (df.get("city_group") == city).astype(int)

    annual = df.groupby(["year_min", "in_city"]).size().unstack(fill_value=0)

    years = list(range(start_year, end_year + 1))
    stats = pd.DataFrame(
        {
            "pub_count": pub_counts.reindex(years, fill_value=0),
            "nyc_count": annual.get(1, pd.Series(dtype=int)).reindex(
                years, fill_value=0
            ),
            "other_count": annual.get(0, pd.Series(dtype=int)).reindex(
                years, fill_value=0
            ),
        },
        index=years,
    )

    return stats


# ---------------------------------------------------------------------------
# Statistical validation functions
# ---------------------------------------------------------------------------


def pearson_summary(x: pd.Series, y: pd.Series):
    """Return N, r, and p‑value (two‑sided) for Pearson correlation."""
    mask = x.notna() & y.notna()
    r, p = pearsonr(x[mask], y[mask])
    n = mask.sum()
    return n, r, p


def ols_with_hac(y: pd.Series, x: pd.Series, hac_lag: int = 1):
    """
    Fit y ~ x + 1, return fitted model, HAC‑robust t‑stats, p‑value, DW.
    """
    mask = y.notna() & x.notna()
    y_, X_ = y[mask], sm.add_constant(x[mask])
    model = sm.OLS(y_, X_).fit()
    hac = model.get_robustcov_results(
        cov_type="HAC", maxlags=hac_lag, use_correction=True
    )
    dw = durbin_watson(model.resid)
    return model, hac, dw


def granger_tests(series_a: pd.Series, series_b: pd.Series, maxlag: int = 5):
    """
    Run bidirectional Granger causality on first differences.
    Returns dict with min p‑value and corresponding lag for each direction.
    """
    # align & difference
    df = pd.concat({"a": series_a, "b": series_b}, axis=1).dropna()
    df = df.diff().dropna()

    # a → b  (does a help predict b?)
    g_ab = grangercausalitytests(df[["b", "a"]], maxlag=maxlag, verbose=False)
    pvals_ab = {lag: res[0]["ssr_ftest"][1] for lag, res in g_ab.items()}
    best_ab = min(pvals_ab, key=pvals_ab.get)

    # b → a
    g_ba = grangercausalitytests(df[["a", "b"]], maxlag=maxlag, verbose=False)
    pvals_ba = {lag: res[0]["ssr_ftest"][1] for lag, res in g_ba.items()}
    best_ba = min(pvals_ba, key=pvals_ba.get)

    return {
        "a_causes_b": {"lag": best_ab, "p": pvals_ab[best_ab]},
        "b_causes_a": {"lag": best_ba, "p": pvals_ba[best_ba]},
    }


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Rolling correlation and statistical validation of the "
        "publisher count versus NYC/non‑NYC imprint counts"
    )
    parser.add_argument(
        "--input-csv", default="./data/PS/data.csv", help="Path to cleaned PS data CSV"
    )
    parser.add_argument(
        "--city", default="New York", help="City treated as in‑city (default: New York)"
    )
    parser.add_argument(
        "--start-year", type=int, default=1900, help="Inclusive start year"
    )
    parser.add_argument("--end-year", type=int, default=2010, help="Inclusive end year")
    parser.add_argument(
        "--window", type=int, default=5, help="Window (yrs) for rolling correlation"
    )
    parser.add_argument(
        "--output",
        default="./viz/ps_pub_loc_correlation.png",
        help="Path for output PNG plot",
    )
    parser.add_argument(
        "--hac-lag", type=int, default=1, help="Max lag for HAC (Newey–West) covariance"
    )
    parser.add_argument(
        "--gc-maxlag", type=int, default=5, help="Maximum lag for Granger causality"
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Load + aggregate
    # ---------------------------------------------------------------------
    df_raw = load_data(args.input_csv, args.start_year, args.end_year)
    stats = compute_annual_counts(df_raw, args.city, args.start_year, args.end_year)

    # ---------------------------------------------------------------------
    # Rolling correlations (unchanged visualisation logic)
    # ---------------------------------------------------------------------
    corr_nyc = (
        stats["pub_count"]
        .rolling(window=args.window, center=True, min_periods=2)
        .corr(stats["nyc_count"])
    )
    corr_other = (
        stats["pub_count"]
        .rolling(window=args.window, center=True, min_periods=2)
        .corr(stats["other_count"])
    )

    # SEs via Fisher z
    n_win = args.window
    if n_win > 3:
        denom = np.sqrt(n_win - 3)
        se_nyc = (1 - corr_nyc**2) / denom
        se_other = (1 - corr_other**2) / denom
    else:
        se_nyc = pd.Series(np.nan, index=stats.index)
        se_other = pd.Series(np.nan, index=stats.index)

    # Plot
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    years = stats.index
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    plt.plot(years, corr_nyc, label=f"Pub vs {args.city}", color="C0")
    plt.plot(years, corr_other, label="Pub vs Other", color="C1")
    plt.fill_between(
        years, corr_nyc - se_nyc, corr_nyc + se_nyc, color="C0", alpha=0.2, linewidth=0
    )
    plt.fill_between(
        years,
        corr_other - se_other,
        corr_other + se_other,
        color="C1",
        alpha=0.2,
        linewidth=0,
    )
    plt.xlabel("Year")
    plt.ylabel("Correlation coefficient")
    plt.title(f"Rolling {args.window}-year correlation")
    plt.legend()
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.savefig(args.output, dpi=600)
    print(f"[INFO] Saved rolling‑correlation plot → {args.output}")

    print("------------------------------------------------------------------")
    print("Rolling correlation summary (means across window centres)")
    print(
        f"  Pub vs {args.city:<12}: r̄ = {corr_nyc.mean():.4f},  SĒ = {se_nyc.mean():.4f}"
    )
    print(
        f"  Pub vs Other         : r̄ = {corr_other.mean():.4f}, SĒ = {se_other.mean():.4f}"
    )
    print("------------------------------------------------------------------\n")

    # ---------------------------------------------------------------------
    # 1. Simple Pearson correlation on full series (pub vs other_count)
    # ---------------------------------------------------------------------
    n, r, p = pearson_summary(stats["pub_count"], stats["other_count"])
    print("[Pearson correlation] publishers  vs  non‑NYC imprints")
    print(f"  N  = {n}")
    print(f"  r  = {r:.4f}")
    print(f"  p  = {p:.4g}")
    print("------------------------------------------------------------------\n")

    # ---------------------------------------------------------------------
    # 2. OLS with Durbin–Watson and HAC‑robust inference
    # ---------------------------------------------------------------------
    model, hac, dw = ols_with_hac(
        stats["other_count"], stats["pub_count"], hac_lag=args.hac_lag
    )

    # position of the explanatory variable in the design matrix
    param_name = "pub_count"
    param_idx = model.model.exog_names.index(param_name)

    coef = hac.params[param_idx]
    tval = hac.tvalues[param_idx]
    p_rob = hac.pvalues[param_idx]

    print("[OLS diagnostics] non‑NYC ~ publishers")
    print(f"  Durbin‑Watson        : {dw:.3f}")
    print(f"  β̂ ({param_name})     : {coef:.4f}")
    print(f"  HAC‑robust t‑stat    : {tval:.3f}")
    print(f"  HAC‑robust p‑value   : {p_rob:.4g}")
    print(f"  (HAC max‑lag = {args.hac_lag})")
    print("------------------------------------------------------------------\n")

    # ---------------------------------------------------------------------
    # 3. Granger causality (first‑differences)
    # ---------------------------------------------------------------------
    gc = granger_tests(stats["pub_count"], stats["other_count"], maxlag=args.gc_maxlag)

    print("[Granger causality] first‑differenced series")
    print(
        f"  publishers  →  non‑NYC : best lag = {gc['a_causes_b']['lag']}, "
        f"p = {gc['a_causes_b']['p']:.4g}"
    )
    print(
        f"  non‑NYC     →  publishers : best lag = {gc['b_causes_a']['lag']}, "
        f"p = {gc['b_causes_a']['p']:.4g}"
    )
    print(f"  (max lag tested = {args.gc_maxlag})")
    print("------------------------------------------------------------------")


if __name__ == "__main__":
    main()
