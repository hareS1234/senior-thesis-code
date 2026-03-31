#!/usr/bin/env python
"""
committor_linear_baseline.py

Non-GNN baselines for committor prediction on node features alone
(no message passing).

Purpose: establish whether the GNN's R² ≈ 0.077 comes from message passing
or from the node features themselves.  If linear regression also gives
R² ≈ 0.07, the node features carry the signal and message passing adds nothing.
If linear gives R² ≈ 0.0 but GAT gives 0.077, then message passing contributes
real (albeit small) value.

Uses the same KTNDataset, train/val split, and evaluation as train_gnn_v2.py,
but replaces the GNN with sklearn models applied independently per node.

Outputs
-------
  {out_dir}/linear_baseline_summary.csv   — R², MAE per model
  {out_dir}/baseline_val_predictions.csv  — validation predictions per model
  {out_dir}/fig_committor_baselines.pdf   — pred vs true scatter per model

Usage:
    python committor_linear_baseline.py \
        --targets-csv GTcheck_micro_vs_coarse_T300K_full.csv \
        --out-dir     linear_baseline_results
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import torch

from ktn_dataset import KTNDataset

warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================
#  Baseline models
# ======================================================================

BASELINES = {
    "LinearRegression": (LinearRegression, {}),
    "Ridge_alpha1": (Ridge, {"alpha": 1.0}),
    "Ridge_alpha10": (Ridge, {"alpha": 10.0}),
    "RandomForest": (RandomForestRegressor, {
        "n_estimators": 100, "max_depth": 8, "random_state": 42, "n_jobs": -1,
    }),
    "GradientBoosting": (GradientBoostingRegressor, {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
        "random_state": 42,
    }),
    "MLP_2layer": (MLPRegressor, {
        "hidden_layer_sizes": (64, 32), "max_iter": 500, "random_state": 42,
        "early_stopping": True, "validation_fraction": 0.15,
    }),
}


def load_gnn_reference(results_dir: Path) -> list[dict]:
    """
    Load GNN reference metrics from metrics_*.json files if present.

    This avoids hard-coding thesis numbers in the baseline script.  If the
    directory is absent or empty, return an empty list and let the non-GNN
    baselines stand on their own.
    """
    if not results_dir.exists():
        return []

    refs: list[dict] = []
    for jf in sorted(results_dir.glob("metrics_*.json")):
        try:
            with open(jf) as f:
                data = json.load(f)
        except Exception:
            continue

        refs.append({
            "family": "GNN reference",
            "model": f"{data.get('config', jf.stem.replace('metrics_', ''))} (GNN)",
            "val_R2": data.get("val_r2"),
            "val_MAE": data.get("val_mae"),
            "n_train": data.get("n_train_nodes", "~"),
            "n_val": data.get("n_val_nodes", "~"),
            "n_features": data.get("node_dim", "same node features"),
            "n_params": data.get("n_params", "N/A"),
        })

    refs.sort(
        key=lambda row: (
            float("-inf")
            if row["val_R2"] is None
            else float(row["val_R2"])
        ),
        reverse=True,
    )
    return refs


# ======================================================================
#  Extract node features and targets with same split as GNN
# ======================================================================

def extract_node_data(
    dataset,
    task: str = "committor",
    train_frac: float = 0.8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract (X_train, y_train, X_val, y_val) pooled across all graphs,
    using the exact same interior-node train/val split as train_gnn_v2.py.

    Returns numpy arrays ready for sklearn.
    """
    target_attr = "committor" if task == "committor" else "mfpt_to_B"

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []

    for graph_idx, data in enumerate(dataset):
        if not hasattr(data, target_attr) or getattr(data, target_attr) is None:
            continue

        x_np = data.x.numpy()  # [N, D_node]
        target_np = getattr(data, target_attr).numpy()  # [N]
        N = x_np.shape[0]

        A_mask = data.A_mask.numpy() if hasattr(data, "A_mask") else np.zeros(N, dtype=bool)
        B_mask = data.B_mask.numpy() if hasattr(data, "B_mask") else np.zeros(N, dtype=bool)

        interior = ~(A_mask | B_mask)
        interior_idx = np.where(interior)[0]

        # Same RNG seeding as train_gnn_v2.py
        rng = np.random.default_rng(seed + 1009 * graph_idx)
        rng.shuffle(interior_idx)

        if len(interior_idx) <= 1:
            split = len(interior_idx)
        else:
            split = int(len(interior_idx) * train_frac)
            split = min(max(split, 1), len(interior_idx) - 1)

        train_idx = np.concatenate([
            np.where(A_mask)[0],
            np.where(B_mask)[0],
            interior_idx[:split],
        ])
        val_idx = interior_idx[split:]

        if len(train_idx) > 0:
            X_train_list.append(x_np[train_idx])
            y_train_list.append(target_np[train_idx])
        if len(val_idx) > 0:
            X_val_list.append(x_np[val_idx])
            y_val_list.append(target_np[val_idx])

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_val = np.vstack(X_val_list)
    y_val = np.concatenate(y_val_list)

    return X_train, y_train, X_val, y_val


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Non-GNN baselines for committor prediction.")
    parser.add_argument("--root", type=str, default="ktn_pyg_data_v2",
                        help="PyG dataset root (reuses GNN cache).")
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--task", type=str, default="committor",
                        choices=["committor", "mfpt"])
    parser.add_argument("--out-dir", type=Path,
                        default=Path("linear_baseline_results"))
    parser.add_argument("--gnn-results-dir", type=Path,
                        default=Path("gnn_results_v2"),
                        help="Directory containing metrics_*.json from GNN runs.")
    parser.add_argument("--scatter-subsample", type=int, default=2000,
                        help="Max number of validation points per panel.")
    parser.add_argument("--max-plot-models", type=int, default=6,
                        help="Maximum number of baseline panels in the PDF.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load same dataset as GNN ───────────────────────────────────────
    from config import BASE_DIR
    print("[baseline] Loading KTN dataset...")
    dataset = KTNDataset(
        root=args.root,
        base_dir=BASE_DIR,
        T=300.0,
        targets_csv=args.targets_csv,
        compute_node_targets=True,
    )
    print(f"[baseline] {len(dataset)} graphs loaded.")

    # ── Extract pooled node features with same split ───────────────────
    print("[baseline] Extracting node features and targets...")
    X_train, y_train, X_val, y_val = extract_node_data(
        dataset, task=args.task, train_frac=0.8, seed=args.seed,
    )
    print(f"[baseline] Train: {X_train.shape[0]:,} nodes, "
          f"Val: {X_val.shape[0]:,} nodes, "
          f"Features: {X_train.shape[1]}")

    # ── Standardize ────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # ── Run all baselines ──────────────────────────────────────────────
    results = []
    prediction_table = pd.DataFrame({"y_true": y_val})

    # Also add constant-prediction baseline (predict mean of train)
    y_mean_pred = np.full_like(y_val, y_train.mean())
    r2_mean = r2_score(y_val, y_mean_pred)
    mae_mean = mean_absolute_error(y_val, y_mean_pred)
    results.append({
        "family": "Node-feature baseline",
        "model": "MeanBaseline",
        "val_R2": r2_mean,
        "val_MAE": mae_mean,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_features": X_train.shape[1],
        "n_params": 0,
    })
    print(f"\n  MeanBaseline: R² = {r2_mean:.4f}, MAE = {mae_mean:.4f}")

    all_val_preds = {"MeanBaseline": y_mean_pred}
    prediction_table["pred_MeanBaseline"] = y_mean_pred

    for name, (model_cls, model_kwargs) in BASELINES.items():
        print(f"\n  Training {name}...")
        try:
            model = model_cls(**model_kwargs)
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)

            # Clip committor predictions to [0, 1]
            if args.task == "committor":
                y_pred = np.clip(y_pred, 0.0, 1.0)

            val_r2 = r2_score(y_val, y_pred)
            val_mae = mean_absolute_error(y_val, y_pred)

            # Count parameters
            if hasattr(model, "coef_"):
                n_params = model.coef_.size + (model.intercept_.size
                                               if hasattr(model.intercept_, "size")
                                               else 1)
            elif hasattr(model, "n_estimators"):
                n_params = f"~{model.n_estimators} trees"
            else:
                n_params = "N/A"

            results.append({
                "family": "Node-feature baseline",
                "model": name,
                "val_R2": val_r2,
                "val_MAE": val_mae,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_features": X_train.shape[1],
                "n_params": n_params,
            })
            all_val_preds[name] = y_pred
            prediction_table[f"pred_{name}"] = y_pred
            print(f"  {name}: R² = {val_r2:.4f}, MAE = {val_mae:.4f}")

        except Exception as e:
            print(f"  {name}: FAILED — {e}")
            results.append({
                "family": "Node-feature baseline",
                "model": name,
                "val_R2": np.nan,
                "val_MAE": np.nan,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_features": X_train.shape[1],
                "n_params": "N/A",
            })

    # ── Add GNN results for direct comparison when available ───────────
    gnn_reference = load_gnn_reference(args.gnn_results_dir)
    results.extend(gnn_reference)

    # ── Save summary ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.out_dir / "linear_baseline_summary.csv", index=False)
    prediction_table.to_csv(args.out_dir / "baseline_val_predictions.csv", index=False)

    # Also save as JSON for easy LaTeX consumption
    with open(args.out_dir / "linear_baseline_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ── Scatter plots ──────────────────────────────────────────────────
    baseline_rows = (
        results_df[results_df["family"] == "Node-feature baseline"]
        .sort_values("val_R2", ascending=False, na_position="last")
    )
    sklearn_models = [
        row.model for row in baseline_rows.itertuples()
        if row.model in all_val_preds and row.model != "MeanBaseline"
    ]
    n_plots = min(len(sklearn_models), args.max_plot_models)
    if n_plots > 0:
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4.5))
        if n_plots == 1:
            axes = [axes]

        for ax, name in zip(axes, sklearn_models[:n_plots]):
            y_pred = all_val_preds[name]
            r2 = r2_score(y_val, y_pred)

            # Subsample for visibility (35k points is too dense)
            rng = np.random.default_rng(42)
            if len(y_val) > args.scatter_subsample:
                idx = rng.choice(len(y_val), size=args.scatter_subsample, replace=False)
            else:
                idx = np.arange(len(y_val))

            ax.scatter(y_val[idx], y_pred[idx], s=1, alpha=0.15, color="steelblue")
            if args.task == "committor":
                lo, hi = 0.0, 1.0
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
            else:
                lo = float(np.nanmin(np.concatenate([y_val[idx], y_pred[idx]])))
                hi = float(np.nanmax(np.concatenate([y_val[idx], y_pred[idx]])))
            ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, alpha=0.5)
            ax.set_xlabel(f"True {args.task}", fontsize=10)
            ax.set_ylabel(f"Predicted {args.task}", fontsize=10)
            ax.set_title(f"{name}\n$R^2$ = {r2:.4f}", fontsize=10)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)

        best_gnn_r2 = max(
            (row["val_R2"] for row in gnn_reference if row.get("val_R2") is not None),
            default=None,
        )
        title_suffix = (
            f" | best GNN $R^2$ = {best_gnn_r2:.3f}"
            if best_gnn_r2 is not None else ""
        )
        fig.suptitle(
            f"Node-feature baselines (no message passing){title_suffix}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()
        fig.savefig(args.out_dir / "fig_committor_baselines.pdf",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

    # ── Print comparison table ─────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMMITTOR PREDICTION: BASELINES vs GNNs")
    print(f"{'='*70}")
    print(results_df.to_string(index=False))
    if gnn_reference:
        best_non_gnn = baseline_rows.iloc[0]
        best_gnn = (
            pd.DataFrame(gnn_reference)
            .sort_values("val_R2", ascending=False, na_position="last")
            .iloc[0]
        )
        delta = float(best_gnn["val_R2"]) - float(best_non_gnn["val_R2"])
        print(
            f"\n  Best non-GNN baseline: {best_non_gnn['model']} "
            f"(R² = {best_non_gnn['val_R2']:.4f})"
        )
        print(
            f"  Best GNN reference:   {best_gnn['model']} "
            f"(R² = {best_gnn['val_R2']:.4f})"
        )
        print(f"  Message-passing gain: {delta:+.4f} R²")
    print(f"\nAll outputs saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
