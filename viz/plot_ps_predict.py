#!/usr/bin/env python3
"""
Generate a line plot showing the percentage of PS-class imprints
published in New York over time, fit a linear regression model,
and predict the percentage for a specified year.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def load_data(csv_path: str) -> pd.DataFrame:
    """Load cleaned imprint data from CSV."""
    return pd.read_csv(csv_path)

def compute_city_share(
    df: pd.DataFrame,
    city: str,
    start_year: int,
    end_year: int,
    window: int,
) -> pd.DataFrame:
    """
    Compute rolling-window percentage of records published in `city`.
    Returns a DataFrame indexed by year with two columns:
    0 = other locations, 1 = city.
    """
    df["in_city"] = (df.get("city_group") == city).astype(int)
    mask = (df.get("year_min") >= start_year) & (df.get("year_min") <= end_year)
    grouped = (
        df[mask]
        .groupby(["year_min", "in_city"])
        .size()
        .unstack(fill_value=0)
        .rolling(window, min_periods=1)
        .mean()
    )
    pct = grouped.div(grouped.sum(axis=1), axis=0) * 100
    return pct

def main():
    parser = argparse.ArgumentParser(
        description="Plot PS imprints share in New York and predict future value"
    )
    parser.add_argument(
        "--input-csv",
        default="./data/PS/data.csv",
        help="Path to cleaned PS data CSV (default: ./data/PS/data.csv)",
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
        "--window",
        type=int,
        default=5,
        help="Rolling window size in years (default: 5)",
    )
    parser.add_argument(
        "--predict-year",
        type=int,
        default=2000,
        help="Year to predict percentage (default: 2000)",
    )
    parser.add_argument(
        "--output",
        default="./viz/ps_predict.png",
        help="Output file path for the figure (default: ./viz/ps_predict.png)",
    )
    args = parser.parse_args()

    df = load_data(args.input_csv)
    pct = compute_city_share(
        df,
        city=args.city,
        start_year=args.start_year,
        end_year=args.end_year,
        window=args.window,
    )
    # Series for city percentage
    pct_city = pct.get(1)
    years = pct_city.index
    # Dynamic fit-end year: year with max city percentage
    fit_end_year = int(pct_city.idxmax())
    print(f"Fitting linear model from {args.start_year} to {fit_end_year}")

    # Prepare data for linear regression
    fit_mask = (years >= args.start_year) & (years <= fit_end_year)
    X = years[fit_mask].values.reshape(-1, 1)
    y = pct_city[fit_mask].values
    mask = ~np.isnan(y)
    X_fit = X[mask]
    y_fit = y[mask]

    # Fit linear model
    model = LinearRegression()
    model.fit(X_fit, y_fit)
    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X_fit, y_fit)
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R^2 score: {r2:.4f}")

    # Predict for target year
    year_pred = np.array([[args.predict_year]])
    y_pred = model.predict(year_pred)
    print(
        f"Predicted percentage of works published in {args.city} in {args.predict_year}: {y_pred[0]:.2f}%"
    )

    # Plot data and model
    plt.figure(dpi=600)
    plt.style.use("tableau-colorblind10")
    plt.plot(years, pct_city, label="LC MDS Data")
    plt.plot(
        X_fit.flatten(),
        model.predict(X_fit),
        label=f"Linear Model ({args.start_year}-{fit_end_year})",
        linestyle="--",
    )
    plt.scatter(
        args.predict_year,
        y_pred,
        color="red",
        label=f"Prediction for {args.predict_year} ({args.start_year}-{fit_end_year})",
        marker="*",
        s=50,
    )
    plt.axhline(y=50, color="gray", linestyle="--", linewidth=1)
    # Annotate R^2 on plot
    ax = plt.gca()
    ax.text(
        0.98,
        0.98,
        f"R² = {r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    plt.title(f"Predicted percentage of PS imprints published in {args.city}")
    plt.xlabel("Year")
    plt.ylabel(f"Percentage in {args.city}")
    plt.legend(title="Published in New York")
    plt.tight_layout()

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        plt.savefig(args.output, dpi=600)
        print(f"Saved figure to: {args.output}")
    else:
        plt.show()

if __name__ == "__main__":
    main()