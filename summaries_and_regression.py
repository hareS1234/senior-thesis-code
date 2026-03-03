"""
summaries_and_regression.py

Quick analysis of the aggregated per-run summaries:

- load all_sequences_summary.csv
- basic sanity checks
- example regressions (e.g. MFPT vs barrier distances)
- save a couple of small CSVs / plots you can drop into the thesis.

You will almost certainly tweak/extend this in thesis_analysis.ipynb.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # conda install scikit-learn

THIS_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = THIS_DIR / "all_sequences_summary.csv"


def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found. Run run_all_sequences.py first."
        )
    df = pd.read_csv(SUMMARY_CSV)
    return df


def simple_regressions(df: pd.DataFrame):
    """
    Example regressions:
    - log(MFPT_AB) vs avg_barrier_AB
    - log(MFPT_AB) vs min_barrier_AB
    - log(MFPT_AB) vs avg_rate_length_AB
    """

    df = df.copy()
    df = df[df["has_AB"]]  # keep only runs with A/B defined
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "mfpt_AB",
            "avg_barrier_AB",
            "min_barrier_AB",
            "avg_rate_length_AB",
        ]
    )

    df["log_mfpt_AB"] = np.log(df["mfpt_AB"])

    X_cols_sets = [
        ["avg_barrier_AB"],
        ["min_barrier_AB"],
        ["avg_rate_length_AB"],
    ]

    results = []

    for cols in X_cols_sets:
        X = df[cols].values.reshape(-1, len(cols))
        y = df["log_mfpt_AB"].values
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = model.score(X, y)
        results.append(
            {
                "features": ",".join(cols),
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
                "r2": float(r2),
            }
        )

        # quick scatter plot
        plt.figure(figsize=(5, 4))
        plt.scatter(X[:, 0], y, s=10, alpha=0.7)
        xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        ys = model.predict(xs.reshape(-1, len(cols)))
        plt.plot(xs, ys, lw=2)
        plt.xlabel(cols[0])
        plt.ylabel("log MFPT_AB")
        plt.title(f"log(MFPT_AB) vs {cols[0]} (R²={r2:.3f})")
        out_png = THIS_DIR / f"regression_mfptAB_vs_{cols[0]}.png"
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()

    results_df = pd.DataFrame(results)
    results_df.to_csv(THIS_DIR / "regression_results.csv", index=False)
    print(results_df)


def main():
    df = load_summary()
    print(f"Loaded {len(df)} rows from {SUMMARY_CSV}")
    simple_regressions(df)


if __name__ == "__main__":
    main()
