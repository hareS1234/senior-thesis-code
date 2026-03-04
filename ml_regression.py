#!/usr/bin/env python
"""
ml_regression.py

Classical ML regression: test whether hand-crafted graph features predict
kinetic observables (MFPTs, relaxation times).

Usage:
    python ml_regression.py \
        --features-csv graph_features_coarse_T300K.csv \
        --targets-csv  GTcheck_micro_vs_coarse_T300K_full.csv \
        --out-dir      ml_results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
from pandas.api.types import is_numeric_dtype

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV,
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)

METADATA_COLS = [
    "dps_dir", "sequence", "system", "variant", "status",
    "coarse_dir", "markov_dir", "T_K",
]

TARGET_DEFS = {
    "log_MFPT_AB": ("MFPT_coarse_AB", np.log10),
    "log_MFPT_BA": ("MFPT_coarse_BA", np.log10),
    "log_t1": ("t1", np.log10),
    "t1_over_t2": ("t1_over_t2", None),
}


# ======================================================================
#  Data loading
# ======================================================================

def load_and_merge_data(
    features_csv: Path,
    targets_csv: Path,
) -> pd.DataFrame:
    """
    Merge graph features with kinetic targets.

    Joins on dps_dir.  Adds log-transformed targets.
    Filters to rows where both features and targets are valid.
    """
    feat_df = pd.read_csv(features_csv)
    tgt_df = pd.read_csv(targets_csv)

    # Normalize dps_dir for joining
    feat_df["dps_dir"] = feat_df["dps_dir"].astype(str).str.rstrip("/")
    tgt_df["dps_dir"] = tgt_df["dps_dir"].astype(str).str.rstrip("/")

    df = feat_df.merge(tgt_df, on="dps_dir", how="inner", suffixes=("", "_tgt"))

    # Keep complete rows and partially-computed rows; exclude hard failures.
    if "status" in df.columns:
        status = df["status"].astype(str)
        df = df[status.eq("OK") | status.str.startswith("PARTIAL")].copy()

    # Add log-transformed targets
    for new_col, (src_col, transform) in TARGET_DEFS.items():
        if src_col in df.columns:
            vals = pd.to_numeric(df[src_col], errors="coerce")
            if transform is not None:
                with np.errstate(divide="ignore", invalid="ignore"):
                    df[new_col] = transform(vals.values)
            else:
                df[new_col] = vals
        else:
            print(f"  [ml_regression] WARNING: target column '{src_col}' not found.")

    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Identify numeric feature columns (exclude metadata and targets)."""
    exclude = set(METADATA_COLS) | set(TARGET_DEFS.keys())
    # Also exclude target source columns and identifiers
    exclude.update(TARGET_DEFS[k][0] for k in TARGET_DEFS)
    exclude.update([
        "GT_valid", "ab_ok", "mfpt_ok", "connectivity_ok", "signs_ok",
        "stationarity_ok", "status_tgt",
    ])

    candidates = []
    for col in df.columns:
        if col in exclude:
            continue
        # Exclude derived log columns, eigenvalue columns, timescale columns,
        # and other target-adjacent columns from the GTcheck CSV
        if col.startswith("log_") or col.startswith("lambda"):
            continue
        if col in ("t1", "t2", "t3", "t4", "t5", "t1_over_t2",
                   "MFPT_coarse_AB", "MFPT_coarse_BA",
                   "MFPT_micro_AB", "MFPT_micro_BA",
                   "N_micro", "N_coarse", "nA_micro", "nA_coarse",
                   "nB_micro", "nB_coarse",
                   "relerr_AB", "relerr_BA", "log10_ratio_AB", "log10_ratio_BA"):
            continue
        if is_numeric_dtype(df[col]):
            # Check it's not all NaN
            if df[col].notna().sum() > df.shape[0] * 0.5:
                candidates.append(col)
    return candidates


# ======================================================================
#  LOO-CV
# ======================================================================

def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Leave-one-out CV with per-fold standardization.

    Returns predictions and metrics dict.
    """
    loo = LeaveOneOut()
    y_pred = np.full_like(y, np.nan)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_class(**model_kwargs)
        try:
            model.fit(X_train_s, y_train)
            y_pred[test_idx] = model.predict(X_test_s)
        except Exception:
            # Keep this fold as NaN so one failing model/fold doesn't abort the run.
            continue

    mask = np.isfinite(y_pred) & np.isfinite(y)
    n_obs = int(mask.sum())
    if n_obs == 0:
        metrics = {"R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0}
        return y_pred, metrics

    y_true_masked = y[mask]
    y_pred_masked = y_pred[mask]
    r2 = float(r2_score(y_true_masked, y_pred_masked)) if n_obs >= 2 else np.nan
    metrics = {
        "R2": r2,
        "RMSE": float(np.sqrt(mean_squared_error(y_true_masked, y_pred_masked))),
        "MAE": float(mean_absolute_error(y_true_masked, y_pred_masked)),
        "n": n_obs,
    }
    return y_pred, metrics


# ======================================================================
#  Model comparison
# ======================================================================

MODELS = {
    "OLS": (LinearRegression, {}),
    "Ridge_1": (Ridge, {"alpha": 1.0}),
    "Ridge_10": (Ridge, {"alpha": 10.0}),
    "RidgeCV": (RidgeCV, {"alphas": np.logspace(-3, 3, 20)}),
    "Lasso_01": (Lasso, {"alpha": 0.1, "max_iter": 10000}),
    "Lasso_1": (Lasso, {"alpha": 1.0, "max_iter": 10000}),
    "LassoCV": (LassoCV, {"max_iter": 10000, "cv": 5}),
    "ElasticNet": (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5, "max_iter": 10000}),
    "ElasticNetCV": (ElasticNetCV, {
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9], "max_iter": 10000, "cv": 5,
    }),
    "RandomForest": (RandomForestRegressor, {
        "n_estimators": 100, "max_depth": 5, "random_state": 42,
    }),
    "GradientBoosting": (GradientBoostingRegressor, {
        "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
        "random_state": 42,
    }),
}


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """Run all models on one target via LOO-CV and return comparison table."""
    results = []
    for name, (cls, kwargs) in MODELS.items():
        _, metrics = run_loocv(X, y, cls, kwargs)
        metrics["model"] = name
        results.append(metrics)
    out = pd.DataFrame(results).set_index("model")
    return out.sort_values("R2", ascending=False, na_position="last")


# ======================================================================
#  Feature importance
# ======================================================================

def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_class=GradientBoostingRegressor,
    model_kwargs: dict = None,
) -> pd.DataFrame:
    """
    Permutation importance on the full dataset (for interpretation).
    """
    if model_kwargs is None:
        model_kwargs = {"n_estimators": 100, "max_depth": 4, "random_state": 42}

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = model_class(**model_kwargs)
    model.fit(X_s, y)

    result = permutation_importance(model, X_s, y, n_repeats=50,
                                    random_state=42, scoring="r2")
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance", ascending=False)
    return imp_df


# ======================================================================
#  Forward stepwise selection
# ======================================================================

def forward_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features: int = 8,
    model_class=Ridge,
    model_kwargs: dict = None,
) -> pd.DataFrame:
    """
    Greedy forward feature selection using LOO-CV R² as criterion.
    """
    if model_kwargs is None:
        model_kwargs = {"alpha": 1.0}

    selected: List[int] = []
    remaining = list(range(X.shape[1]))
    history = []

    for step in range(min(max_features, X.shape[1])):
        best_r2 = -np.inf
        best_feat = None

        for f in remaining:
            candidate = selected + [f]
            X_sub = X[:, candidate]
            _, metrics = run_loocv(X_sub, y, model_class, model_kwargs)
            if metrics["R2"] > best_r2:
                best_r2 = metrics["R2"]
                best_feat = f

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        history.append({
            "step": step + 1,
            "feature": feature_names[best_feat],
            "R2": best_r2,
            "features_so_far": ", ".join(feature_names[i] for i in selected),
        })

        # Stop if R² is decreasing
        if step > 0 and best_r2 < history[-2]["R2"] - 0.01:
            break

    return pd.DataFrame(history)


# ======================================================================
#  Plotting
# ======================================================================

def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    target_name: str,
    model_name: str,
    out_path: Path,
):
    """Predicted vs actual scatter with identity line and labels."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    r2 = r2_score(yt, yp)

    ax.scatter(yt, yp, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)

    # Label points
    for i, (x, y) in enumerate(zip(yt, yp)):
        idx = np.where(mask)[0][i]
        ax.annotate(labels[idx], (x, y), fontsize=6, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.3, linewidth=1)
    ax.set_xlim(lo - margin, hi + margin)
    ax.set_ylim(lo - margin, hi + margin)

    ax.set_xlabel(f"Actual {target_name}", fontsize=12)
    ax.set_ylabel(f"Predicted {target_name}", fontsize=12)
    ax.set_title(f"{model_name}  (LOO-CV R² = {r2:.3f})", fontsize=13)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    imp_df: pd.DataFrame,
    target_name: str,
    out_path: Path,
    top_n: int = 15,
):
    """Horizontal bar chart of top feature importances."""
    df = imp_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(1, 1, figsize=(8, 0.4 * len(df) + 1.5))
    ax.barh(df["feature"], df["importance"],
            xerr=df["importance_std"], color="steelblue", alpha=0.8)
    ax.set_xlabel("Permutation Importance (R² decrease)", fontsize=11)
    ax.set_title(f"Feature Importance for {target_name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_forward_selection(
    sel_df: pd.DataFrame,
    target_name: str,
    out_path: Path,
):
    """R² vs number of features plot."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    ax.plot(sel_df["step"], sel_df["R2"], "o-", color="steelblue",
            markersize=8, linewidth=2)
    for _, row in sel_df.iterrows():
        ax.annotate(row["feature"], (row["step"], row["R2"]),
                    fontsize=7, rotation=15, ha="left",
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Number of Features", fontsize=12)
    ax.set_ylabel("LOO-CV R²", fontsize=12)
    ax.set_title(f"Forward Selection: {target_name}", fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classical ML regression on graph features vs kinetic targets."
    )
    parser.add_argument(
        "--features-csv", type=Path, required=True,
        help="CSV from graph_features.py",
    )
    parser.add_argument(
        "--targets-csv", type=Path,
        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"),
        help="CSV from analyze_micro_vs_coarse_T300K.py",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("ml_results"),
        help="Output directory for results.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge
    print("[ml_regression] Loading and merging data...")
    df = load_and_merge_data(args.features_csv, args.targets_csv)
    print(f"[ml_regression] {len(df)} networks with both features and targets.")

    feature_cols = get_feature_cols(df)
    print(f"[ml_regression] Using {len(feature_cols)} features.")
    if not feature_cols:
        print("[ml_regression] No usable numeric features found. Exiting.")
        return

    targets = [t for t in TARGET_DEFS if t in df.columns]
    if not targets:
        print("[ml_regression] No target columns available after merge. Exiting.")
        return

    for target in targets:
        # Keep as many samples as possible for this specific target.
        df_target = df.dropna(subset=feature_cols + [target]).copy()
        print(f"[ml_regression] {target}: {len(df_target)} networks after dropping NaNs.")
        if len(df_target) < 10:
            print(f"  [ml_regression] Skipping {target}: too few samples.")
            continue

        X = df_target[feature_cols].values.astype(float)
        y = df_target[target].values.astype(float)
        if np.isnan(y).all():
            print(f"  [ml_regression] Skipping {target}: all NaN.")
            continue
        labels = df_target["sequence"].tolist() if "sequence" in df_target.columns else \
                 [str(i) for i in range(len(df_target))]

        print(f"\n{'='*60}")
        print(f"  Target: {target}  (N = {len(y)})")
        print(f"{'='*60}")

        # 1) Model comparison
        comp = compare_models(X, y, feature_cols)
        comp.to_csv(args.out_dir / f"model_comparison_{target}.csv")
        print(comp.to_string())

        # 2) Feature importance
        imp = compute_feature_importance(X, y, feature_cols)
        imp.to_csv(args.out_dir / f"feature_importance_{target}.csv", index=False)
        plot_feature_importance(imp, target,
                                args.out_dir / f"feature_importance_{target}.png")

        # 3) Forward selection
        sel = forward_selection(X, y, feature_cols)
        sel.to_csv(args.out_dir / f"forward_selection_{target}.csv", index=False)
        plot_forward_selection(sel, target,
                               args.out_dir / f"forward_selection_{target}.png")

        # 4) Best model predicted vs actual plot + predictions CSV
        valid_r2 = comp["R2"].dropna()
        if valid_r2.empty:
            print("  [ml_regression] Skipping best-model plot: all model R² are NaN.")
            continue

        best_model_name = valid_r2.idxmax()
        best_cls, best_kwargs = MODELS[best_model_name]
        y_pred, _ = run_loocv(X, y, best_cls, best_kwargs)
        plot_predicted_vs_actual(
            y, y_pred, labels, target, best_model_name,
            args.out_dir / f"pred_vs_actual_{target}.png",
        )

        # Save predictions CSV for notebook consumption
        pred_df = pd.DataFrame({
            "dps_dir": df_target["dps_dir"].values,
            "actual": y,
            "predicted": y_pred,
        })
        pred_df.to_csv(args.out_dir / f"predictions_{target}.csv", index=False)

        print(f"  Best model: {best_model_name} (R² = {comp.loc[best_model_name, 'R2']:.3f})")

    # Aggregate model comparison across all targets into one CSV
    all_comp_files = sorted(args.out_dir.glob("model_comparison_*.csv"))
    if all_comp_files:
        dfs = []
        for f in all_comp_files:
            tgt = f.stem.replace("model_comparison_", "")
            comp_df = pd.read_csv(f, index_col=0)
            comp_df["target"] = tgt
            dfs.append(comp_df)
        combined = pd.concat(dfs, ignore_index=False)
        combined.to_csv(args.out_dir / "model_comparison.csv")

    print(f"\n[ml_regression] Results saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
