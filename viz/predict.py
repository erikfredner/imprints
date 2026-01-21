#!/usr/bin/env python3
"""Predict PS-class imprint share in a city and visualize the trend."""
import os
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data/PS/data.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "predict.png"

def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)

def compute_city_share(
    df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Compute annual percentage of records published in `city` using raw counts.
    Returns a DataFrame indexed by year with two columns: 0 = other locations, 1 = city.
    """
    df = df.copy()
    df["in_city"] = (df.get("city_group") == city).astype(int)
    mask = df.get("year_min").between(start_year, end_year)
    grouped = (
        df[mask]
        .groupby(["year_min", "in_city"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    pct.index = pct.index.astype(int)
    return pct

def main():
    parser = argparse.ArgumentParser(
        description="Plot PS imprints share in New York and predict future value"
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to cleaned PS data CSV (default: data/PS/data.csv)",
    )
    parser.add_argument(
        "--city",
        default="New York",
        help="City to highlight (default: New York)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1900,
        help="Start year (inclusive) for analysis (default: 1900)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2010,
        help="End year (inclusive) for analysis (default: 2010)",
    )
    parser.add_argument(
        "--predict-year",
        type=int,
        default=2000,
        help="Year to predict percentage (default: 2000)", 
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output file path for the figure (default: ./viz/ps_predict.png)",
    )
    args = parser.parse_args()

    city_label = (
        "New York City"
        if str(args.city).strip().lower() in {"new york", "nyc"}
        else args.city
    )

    df = load_data(args.input_csv)
    pct = compute_city_share(
        df,
        city=city_label,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    pct_city = pct.get(1)
    years = pct_city.index
    pct_sum = pct.sum(axis=1)
    if np.allclose(pct_sum.dropna(), 100.0, atol=1e-6):
        print("Sanity check: annual location percentages sum to ~100%.")
    else:
        print("Warning: Percentages do not sum to 100 for some years; review input data.")

    fit_df = (
        pd.DataFrame({"year": years, "pct_city": pct_city.values})
        .query("year >= @args.start_year and year <= @args.end_year")
        .dropna()
    )
    if fit_df.empty:
        raise ValueError(
            f"Need at least two years of data between {args.start_year} and {args.end_year} "
            f"for city='{city_label}'. Found {len(fit_df)}. Try widening the window or "
            "adjusting the city name."
        )

    peak_idx = fit_df["pct_city"].idxmax()
    peak_year = int(fit_df.loc[peak_idx, "year"])
    fit_df = fit_df[fit_df["year"] <= peak_year]

    if len(fit_df) < 2:
        raise ValueError(
            f"Need at least two years of data up to the peak year ({peak_year}) to fit a trend. "
            f"Found {len(fit_df)}. Try lowering --start-year or widening the window."
        )

    fit_year_start = int(fit_df["year"].min())
    fit_year_end = peak_year
    print(f"Peak share year: {peak_year} (value: {fit_df.loc[peak_idx, 'pct_city']:.2f}%)")
    print(f"Fitting linear model from {fit_year_start} to {fit_year_end} (upward trend segment)")

    # Handle constant series separately to avoid statsmodels warnings
    is_constant = fit_df["pct_city"].nunique() == 1
    if is_constant:
        constant_value = float(fit_df["pct_city"].iloc[0])
        slope = 0.0
        intercept = constant_value
        r2 = float("nan")
        y_pred = constant_value
        ci_lower = constant_value
        ci_upper = constant_value
        fit_line = pd.Series(constant_value, index=fit_df["year"])
        print(
            "City share is constant in the fit window; slope fixed at 0 and CI collapses to a point."
        )
    else:
        X_fit = sm.add_constant(fit_df["year"], has_constant="add")
        model = sm.OLS(fit_df["pct_city"], X_fit).fit()
        slope = model.params["year"]
        intercept = model.params["const"]
        r2 = model.rsquared
        residual_se = model.mse_resid ** 0.5
        print(f"Slope: {slope:.4f}")
        print(f"Intercept: {intercept:.4f}")
        print(f"R^2 score: {r2:.4f}")
        print(f"Residual standard error: {residual_se:.4f} on {model.df_resid:.0f} dof")

        predict_exog = sm.add_constant(
            pd.Series([args.predict_year], name="year"), has_constant="add"
        )
        prediction = model.get_prediction(predict_exog).summary_frame(alpha=0.05)
        y_pred = prediction["mean"].iloc[0]
        ci_lower = prediction["mean_ci_lower"].iloc[0]
        ci_upper = prediction["mean_ci_upper"].iloc[0]
        fit_line = model.predict(X_fit)

    print(
        "Predicted percentage of works published in "
        f"{city_label} in {args.predict_year}: {y_pred:.2f}% "
        f"(95% CI: {ci_lower:.2f}%, {ci_upper:.2f}%)"
    )

    observed_value = float(pct_city.get(args.predict_year)) if args.predict_year in pct_city.index else float("nan")
    if np.isnan(observed_value):
        print(f"Observed percentage in {args.predict_year}: unavailable (no data for that year).")
    else:
        print(f"Observed percentage in {args.predict_year}: {observed_value:.2f}%")

    # Plot data and model
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    plt.plot(years, pct_city, label="LC MDS Data")
    plt.plot(
        fit_df["year"],
        fit_line,
        label=f"Linear Model ({fit_year_start}-{fit_year_end})",
        linestyle="--",
    )
    plt.scatter(
        args.predict_year,
        y_pred,
        color="red",
        label=f"Prediction for {args.predict_year} ({fit_year_start}-{fit_year_end})",
        marker="*",
        s=50,
    )
    if not np.isnan(observed_value):
        plt.scatter(
            args.predict_year,
            observed_value,
            color="black",
            label=f"Observed {args.predict_year}",
            marker="o",
            s=35,
            zorder=5,
        )
    plt.errorbar(
        args.predict_year,
        y_pred,
        yerr=[[y_pred - ci_lower], [ci_upper - y_pred]],
        fmt="none",
        ecolor="red",
        elinewidth=1,
        capsize=4,
        label="95% CI for prediction",
    )
    plt.axhline(y=50, color="gray", linestyle="--", linewidth=1)
    # Annotate R^2 on plot
    ax = plt.gca()
    ax.text(
        0.98,
        0.98,
        f"R² = {r2:.3f}" if not np.isnan(r2) else "R² = n/a",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    plt.title(f"Predicted percentage of PS imprints published in {city_label}")
    plt.xlabel("Year")
    plt.ylabel(f"Percentage in {city_label}")
    plt.legend(title=f"Published in {city_label}")
    plt.tight_layout()

    if args.output:
        output_dir = args.output.parent
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(args.output, dpi=600)
        print(f"Saved figure to: {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
