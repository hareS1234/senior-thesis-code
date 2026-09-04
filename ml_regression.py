#!/usr/bin/env python
"""Test how well the hand-built graph features predict the kinetic targets."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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






def load_and_merge_data(
    features_csv: Path,
    targets_csv: Path,
) -> pd.DataFrame:
    """Join features and targets on ``dps_dir`` and add the log targets."""
    feat_df = pd.read_csv(features_csv)
    tgt_df = pd.read_csv(targets_csv)




    feat_df["dps_dir"] = feat_df["dps_dir"].astype(str).str.rstrip("/")
    tgt_df["dps_dir"] = tgt_df["dps_dir"].astype(str).str.rstrip("/")


    feat_df["_join_key"] = feat_df["dps_dir"].apply(lambda p: Path(p).name)
    tgt_df["_join_key"] = tgt_df["dps_dir"].apply(lambda p: Path(p).name)

    df = feat_df.merge(tgt_df, on="_join_key", how="inner", suffixes=("", "_tgt"))
    df.drop(columns=["_join_key", "dps_dir_tgt"], inplace=True, errors="ignore")





    if "status" in df.columns:
        status = df["status"].astype(str)
        df = df[status.eq("OK") | status.str.startswith("PARTIAL")].copy()


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
    """Pick the numeric inputs, leaving out metadata and targets."""
    exclude = set(METADATA_COLS) | set(TARGET_DEFS.keys())

    exclude.update(TARGET_DEFS[k][0] for k in TARGET_DEFS)
    exclude.update([
        "GT_valid", "ab_ok", "mfpt_ok", "connectivity_ok", "signs_ok",
        "stationarity_ok", "status_tgt",
    ])

    candidates = []
    for col in df.columns:
        if col in exclude:
            continue


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

            if df[col].notna().sum() > df.shape[0] * 0.5:
                candidates.append(col)
    return candidates






def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run LOO-CV with imputation and scaling fitted inside each fold."""
    loo = LeaveOneOut()
    y_pred = np.full_like(y, np.nan)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X_train_s = pipe.fit_transform(X_train)
        X_test_s = pipe.transform(X_test)

        model = model_class(**model_kwargs)
        try:
            model.fit(X_train_s, y_train)
            y_pred[test_idx] = model.predict(X_test_s)
        except Exception:

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
    model_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare the selected models on one target."""
    selected = model_names if model_names is not None else list(MODELS)
    results = []
    for name in selected:
        if name not in MODELS:
            print(f"  [ml_regression] WARNING: unknown model '{name}', skipping.")
            continue
        cls, kwargs = MODELS[name]
        _, metrics = run_loocv(X, y, cls, kwargs)
        metrics["model"] = name
        results.append(metrics)
    if not results:
        return pd.DataFrame(columns=["R2", "RMSE", "MAE", "n"])
    out = pd.DataFrame(results).set_index("model")
    return out.sort_values("R2", ascending=False, na_position="last")






def compute_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    model_class=GradientBoostingRegressor,
    model_kwargs: dict = None,
    n_repeats: int = 50,
) -> pd.DataFrame:
    """Fit once on the full set and get permutation importance."""
    if model_kwargs is None:
        model_kwargs = {"n_estimators": 100, "max_depth": 4, "random_state": 42}

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_s = pipe.fit_transform(X)

    model = model_class(**model_kwargs)
    model.fit(X_s, y)

    result = permutation_importance(
        model,
        X_s,
        y,
        n_repeats=n_repeats,
        random_state=42,
        scoring="r2",
    )
    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance", ascending=False)
    return imp_df






def forward_selection(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    max_features: int = 8,
    model_class=Ridge,
    model_kwargs: dict = None,
) -> pd.DataFrame:
    """Add features greedily using LOO R²."""
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


        if step > 0 and best_r2 < history[-2]["R2"] - 0.01:
            break

    return pd.DataFrame(history)






def plot_predicted_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    target_name: str,
    model_name: str,
    out_path: Path,
):
    """Plot predictions against the real values."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    r2 = r2_score(yt, yp)

    ax.scatter(yt, yp, s=40, alpha=0.7, edgecolors="k", linewidths=0.5)


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
    """Plot the strongest permutation scores."""
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
    """Plot R² as the selected feature set grows."""
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
    parser.add_argument(
        "--impute", action="store_true",
        help="Impute missing feature values (median) instead of requiring complete features.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=None,
        help=f"Subset of targets to run. Choices: {', '.join(TARGET_DEFS.keys())}",
    )
    parser.add_argument(
        "--models", nargs="+", default=list(MODELS.keys()),
        help=f"Subset of models to compare. Choices: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--n-perm-repeats", type=int, default=50,
        help="Permutation-importance repeats (default: 50). Lower this for CPU-light runs.",
    )
    parser.add_argument(
        "--skip-forward-selection", action="store_true",
        help="Skip greedy forward selection to save time.",
    )
    parser.add_argument(
        "--forward-max-features", type=int, default=8,
        help="Maximum number of features in forward selection (default: 8).",
    )
    parser.add_argument(
        "--min-samples", type=int, default=10,
        help="Minimum usable samples required to analyze a target (default: 10).",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[ml_regression] Loading and merging data...")
    df = load_and_merge_data(args.features_csv, args.targets_csv)
    print(f"[ml_regression] {len(df)} networks with both features and targets.")

    feature_cols = get_feature_cols(df)
    print(f"[ml_regression] Using {len(feature_cols)} features.")
    if not feature_cols:
        print("[ml_regression] No usable numeric features found. Exiting.")
        return

    available_targets = [t for t in TARGET_DEFS if t in df.columns]
    if not available_targets:
        print("[ml_regression] No target columns available after merge. Exiting.")
        return

    if args.targets is None:
        targets = available_targets
    else:
        targets = [t for t in args.targets if t in available_targets]
        missing_targets = [t for t in args.targets if t not in available_targets]
        for t in missing_targets:
            print(f"[ml_regression] WARNING: requested target '{t}' not available after merge.")
    if not targets:
        print("[ml_regression] No requested targets available. Exiting.")
        return

    model_names = [m for m in args.models if m in MODELS]
    missing_models = [m for m in args.models if m not in MODELS]
    for m in missing_models:
        print(f"[ml_regression] WARNING: unknown model '{m}', skipping.")
    if not model_names:
        print("[ml_regression] No valid models selected. Exiting.")
        return

    print(f"[ml_regression] Targets: {', '.join(targets)}")
    print(f"[ml_regression] Models: {', '.join(model_names)}")
    print(f"[ml_regression] Permutation repeats: {args.n_perm_repeats}")
    print(f"[ml_regression] Forward selection: {'off' if args.skip_forward_selection else 'on'}")

    summary_rows = []

    for target in targets:
        df_target = df.dropna(subset=[target]).copy()

        X = df_target[feature_cols].values.astype(float)
        y = df_target[target].values.astype(float)

        if not args.impute:
            keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            df_target = df_target.loc[keep].copy()
            X = X[keep]
            y = y[keep]
            print(f"[ml_regression] {target}: {len(df_target)} networks (strict, NaN rows dropped).")
        else:
            print(f"[ml_regression] {target}: {len(df_target)} networks (--impute, NaN features filled).")

        if len(df_target) < args.min_samples:
            print(f"  [ml_regression] Skipping {target}: too few samples (< {args.min_samples}).")
            continue
        if np.isnan(y).all():
            print(f"  [ml_regression] Skipping {target}: all NaN.")
            continue

        labels = df_target["sequence"].tolist() if "sequence" in df_target.columns else [str(i) for i in range(len(df_target))]

        print(f"\n{'='*60}")
        print(f"  Target: {target}  (N = {len(y)})")
        print(f"\n{'='*60}")

        comp = compare_models(X, y, feature_cols, model_names=model_names)
        if comp.empty:
            print(f"  [ml_regression] No successful models for {target}.")
            continue
        comp.to_csv(args.out_dir / f"model_comparison_{target}.csv")
        print(comp.to_string())

        valid_r2 = comp["R2"].dropna()
        if valid_r2.empty:
            print("  [ml_regression] Skipping downstream analysis: all model R² are NaN.")
            continue

        best_model_name = valid_r2.idxmax()
        best_cls, best_kwargs = MODELS[best_model_name]

        try:
            imp = compute_feature_importance(
                X,
                y,
                feature_cols,
                model_class=best_cls,
                model_kwargs=best_kwargs,
                n_repeats=args.n_perm_repeats,
            )
            imp.to_csv(args.out_dir / f"feature_importance_{target}.csv", index=False)
            plot_feature_importance(
                imp,
                target,
                args.out_dir / f"feature_importance_{target}.png",
            )
        except Exception as e:
            print(f"  [ml_regression] Feature importance failed for {target}: {type(e).__name__}: {e}")

        if not args.skip_forward_selection:
            try:
                sel = forward_selection(
                    X,
                    y,
                    feature_cols,
                    max_features=args.forward_max_features,
                )
                sel.to_csv(args.out_dir / f"forward_selection_{target}.csv", index=False)
                if not sel.empty:
                    plot_forward_selection(
                        sel,
                        target,
                        args.out_dir / f"forward_selection_{target}.png",
                    )
            except Exception as e:
                print(f"  [ml_regression] Forward selection failed for {target}: {type(e).__name__}: {e}")

        y_pred, metrics = run_loocv(X, y, best_cls, best_kwargs)
        plot_predicted_vs_actual(
            y,
            y_pred,
            labels,
            target,
            best_model_name,
            args.out_dir / f"pred_vs_actual_{target}.png",
        )

        pred_df = pd.DataFrame({
            "dps_dir": df_target["dps_dir"].values,
            "actual": y,
            "predicted": y_pred,
        })
        pred_df.to_csv(args.out_dir / f"predictions_{target}.csv", index=False)

        summary_rows.append({
            "target": target,
            "n_samples": len(y),
            "best_model": best_model_name,
            "R2": metrics["R2"],
            "RMSE": metrics["RMSE"],
            "MAE": metrics["MAE"],
            "impute": args.impute,
        })
        pd.DataFrame(summary_rows).to_csv(args.out_dir / "summary.csv", index=False)

        print(f"  Best model: {best_model_name} (R² = {comp.loc[best_model_name, 'R2']:.3f})")

    if summary_rows:
        print(f"\n[ml_regression] Wrote summary for {len(summary_rows)} targets to {args.out_dir / 'summary.csv'}")
    else:
        print("\n[ml_regression] No target completed successfully.")


if __name__ == "__main__":
    main()
