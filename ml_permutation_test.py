#!/usr/bin/env python
"""
ml_permutation_test.py

Permutation test for classical ML regression results.

For each target, shuffles the target labels N_PERM times, reruns the full
LOO-CV pipeline for the best model, and records the null-distribution R².
Compares against the observed R² to compute an empirical p-value:

    p = (# null R² >= observed R² + 1) / (N_PERM + 1)

Also computes bootstrap 95% CIs on the observed LOO predictions for all
four targets, not just MFPT_AB.

Outputs
-------
  {out_dir}/permutation_null_{target}.csv    — null R² values per shuffle
  {out_dir}/permutation_summary.csv          — observed R², p-value, CI
  {out_dir}/observed_predictions_{target}.csv — observed vs predicted pairs
  {out_dir}/fig_permutation_null_{target}.pdf — null distribution + obs line
  {out_dir}/bootstrap_ci.csv                 — bootstrap CIs for all targets

Usage (cluster):
    python ml_permutation_test.py \
        --features-csv graph_features_coarse_T300K_lite.csv \
        --targets-csv  GTcheck_micro_vs_coarse_T300K_full.csv \
        --out-dir      permutation_results \
        --n-perm 1000 \
        --n-bootstrap 2000 \
        --seed 42
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

# ── Reuse the exact data-loading and LOO-CV logic from ml_regression.py ──
from ml_regression import (
    load_and_merge_data,
    get_feature_cols,
    run_loocv,
    TARGET_DEFS,
    METADATA_COLS,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
#  Best-model mapping (must match Table 5 in the thesis)
# ======================================================================

BEST_MODELS = {
    "log_MFPT_AB": (ElasticNet, {"alpha": 0.5, "l1_ratio": 0.5, "max_iter": 10000}),
    "log_MFPT_BA": (GradientBoostingRegressor, {
        "n_estimators": 50, "max_depth": 3, "learning_rate": 0.1,
        "random_state": 42,
    }),
    "log_t1": (RandomForestRegressor, {
        "n_estimators": 100, "max_depth": 5, "random_state": 42,
    }),
    "t1_over_t2": (RandomForestRegressor, {
        "n_estimators": 100, "max_depth": 5, "random_state": 42,
    }),
}

MODEL_LABELS = {
    "log_MFPT_AB": "ElasticNet",
    "log_MFPT_BA": "GradientBoosting",
    "log_t1": "RandomForest",
    "t1_over_t2": "RandomForest",
}


# ======================================================================
#  Permutation test
# ======================================================================

def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_kwargs: dict,
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, dict, np.ndarray, float]:
    """
    Run a permutation test on the LOO-CV R².

    Returns
    -------
    y_pred_obs    : ndarray of observed LOO predictions
    metrics_obs   : metrics dict for observed predictions
    null_r2s      : ndarray of shape (n_perm,)
    p_value       : float (one-sided, upper tail)
    """
    # Observed R²
    y_pred_obs, metrics_obs = run_loocv(X, y, model_class, model_kwargs)
    observed_r2 = metrics_obs["R2"]

    rng = np.random.default_rng(seed)
    null_r2s = np.full(n_perm, np.nan)

    for i in range(n_perm):
        y_shuffled = rng.permutation(y)
        _, metrics_shuf = run_loocv(X, y_shuffled, model_class, model_kwargs)
        null_r2s[i] = metrics_shuf["R2"]
        if (i + 1) % 100 == 0:
            print(f"    permutation {i+1}/{n_perm} done")

    # Empirical p-value (conservative: +1 in numerator and denominator)
    n_ge = np.sum(null_r2s >= observed_r2)
    p_value = (n_ge + 1) / (n_perm + 1)

    return y_pred_obs, metrics_obs, null_r2s, p_value


# ======================================================================
#  Bootstrap confidence intervals
# ======================================================================

def bootstrap_r2_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI on R² from observed vs predicted arrays.

    Returns
    -------
    median_r2, ci_lo, ci_hi
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)

    rng = np.random.default_rng(seed)
    boot_r2s = np.full(n_bootstrap, np.nan)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt_b, yp_b = yt[idx], yp[idx]
        # Need variance in both arrays for a valid R²
        if yt_b.std() > 0:
            boot_r2s[b] = r2_score(yt_b, yp_b)

    valid = boot_r2s[np.isfinite(boot_r2s)]
    alpha = (1 - ci) / 2
    ci_lo = float(np.percentile(valid, 100 * alpha))
    ci_hi = float(np.percentile(valid, 100 * (1 - alpha)))
    median_r2 = float(np.median(valid))

    return median_r2, ci_lo, ci_hi


# ======================================================================
#  Plotting
# ======================================================================

def plot_null_distribution(
    null_r2s: np.ndarray,
    observed_r2: float,
    p_value: float,
    target_name: str,
    out_path: Path,
):
    """Histogram of null R² with observed value marked."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))

    valid = null_r2s[np.isfinite(null_r2s)]
    ax.hist(valid, bins=50, color="steelblue", alpha=0.7, edgecolor="white",
            linewidth=0.5, density=True, label="Null distribution")
    ax.axvline(observed_r2, color="crimson", linewidth=2, linestyle="--",
               label=f"Observed $R^2$ = {observed_r2:.3f}")
    if valid.size:
        ax.axvline(np.median(valid), color="black", linewidth=1.2, linestyle=":",
                   label=f"Null median = {np.median(valid):.3f}")

    ax.set_xlabel("$R^2$ (LOO-CV)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Permutation test: {target_name}  ($p$ = {p_value:.4f})",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Permutation test and bootstrap CIs for ML regression results."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("permutation_results"))
    parser.add_argument("--n-perm", type=int, default=1000,
                        help="Number of permutation shuffles (default: 1000)")
    parser.add_argument("--n-bootstrap", type=int, default=2000,
                        help="Number of bootstrap resamples for CI (default: 2000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Subset of targets. Default: all four.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data (same pipeline as ml_regression.py) ──────────────────
    print("[perm_test] Loading data...")
    df = load_and_merge_data(args.features_csv, args.targets_csv)
    feature_cols = get_feature_cols(df)
    print(f"[perm_test] {len(df)} samples, {len(feature_cols)} features")

    targets = args.targets or list(BEST_MODELS.keys())
    summary_rows = []
    bootstrap_rows = []

    for target in targets:
        if target not in TARGET_DEFS:
            print(f"[perm_test] Unknown target '{target}', skipping.")
            continue

        if target not in df.columns or df[target].isna().all():
            print(f"[perm_test] Target '{target}' missing from data, skipping.")
            continue

        mask = df[target].notna()
        X = df.loc[mask, feature_cols].values.astype(float)
        y = df.loc[mask, target].values.astype(float)

        if len(y) < 10:
            print(f"[perm_test] Only {len(y)} samples for '{target}', skipping.")
            continue

        model_cls, model_kwargs = BEST_MODELS[target]
        print(f"\n{'='*60}")
        print(f"  Target: {target}  (n={len(y)}, model={model_cls.__name__})")
        print(f"{'='*60}")

        # ── 1. Permutation test ────────────────────────────────────────
        print(f"  Running {args.n_perm} permutations...")
        y_pred_obs, metrics_obs, null_r2s, p_value = permutation_test(
            X, y, model_cls, model_kwargs,
            n_perm=args.n_perm, seed=args.seed,
        )
        observed_r2 = metrics_obs["R2"]
        print(f"  Observed R² = {observed_r2:.4f}")
        print(f"  p-value     = {p_value:.4f}")

        # Save null distribution
        null_df = pd.DataFrame({"null_r2": null_r2s})
        null_df.to_csv(
            args.out_dir / f"permutation_null_{target}.csv", index=False
        )

        # Plot
        plot_null_distribution(
            null_r2s, observed_r2, p_value, target,
            args.out_dir / f"fig_permutation_null_{target}.pdf",
        )

        # ── 2. Bootstrap CI on observed predictions ────────────────────
        print(f"  Running {args.n_bootstrap} bootstrap resamples...")
        median_r2, ci_lo, ci_hi = bootstrap_r2_ci(
            y, y_pred_obs, n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
        print(f"  Bootstrap median R² = {median_r2:.4f}  "
              f"95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")

        obs_pred_df = df.loc[mask, [c for c in METADATA_COLS if c in df.columns]].copy()
        obs_pred_df["target"] = target
        obs_pred_df["y_true"] = y
        obs_pred_df["y_pred"] = y_pred_obs
        obs_pred_df.to_csv(
            args.out_dir / f"observed_predictions_{target}.csv", index=False
        )

        null_valid = null_r2s[np.isfinite(null_r2s)]

        summary_rows.append({
            "target": target,
            "model": MODEL_LABELS.get(target, model_cls.__name__),
            "n_samples": len(y),
            "observed_R2": observed_r2,
            "p_value": p_value,
            "n_permutations": args.n_perm,
            "null_median_R2": float(np.median(null_valid)) if len(null_valid) else np.nan,
            "null_mean_R2": float(np.mean(null_valid)) if len(null_valid) else np.nan,
            "null_std_R2": float(np.std(null_valid, ddof=1)) if len(null_valid) > 1 else np.nan,
            "null_q95_R2": float(np.percentile(null_valid, 95)) if len(null_valid) else np.nan,
            "bootstrap_median_R2": median_r2,
            "bootstrap_CI_lo": ci_lo,
            "bootstrap_CI_hi": ci_hi,
            "n_bootstrap": args.n_bootstrap,
        })

        bootstrap_rows.append({
            "target": target,
            "median_R2": median_r2,
            "CI_95_lo": ci_lo,
            "CI_95_hi": ci_hi,
        })

    # ── Save summary ───────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_dir / "permutation_summary.csv", index=False)

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    bootstrap_df.to_csv(args.out_dir / "bootstrap_ci.csv", index=False)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(summary_df.to_string(index=False))
    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
