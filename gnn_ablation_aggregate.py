#!/usr/bin/env python
"""
gnn_ablation_aggregate.py

Collect all per-config JSON files from the GNN ablation sweep into a
single summary CSV and generate a heatmap of R² vs hyperparameters.

Run this AFTER all SLURM array tasks from run_gnn_ablation.sbatch complete.

Outputs
-------
  {out_dir}/gnn_ablation_summary.csv           — full results table
  {out_dir}/fig_gnn_ablation_heatmap.pdf       — heatmaps: R² by (k, h, L)
  {out_dir}/fig_gnn_ablation_best_vs_worst.pdf — bar chart of top and bottom configs

Usage:
    python gnn_ablation_aggregate.py --results-dir gnn_ablation_results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate GNN ablation sweep results.")
    parser.add_argument("--results-dir", type=Path,
                        default=Path("gnn_ablation_results"))
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (defaults to results-dir).")
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect JSON files ─────────────────────────────────────────────
    json_files = sorted(args.results_dir.glob("metrics_gat_*.json"))
    print(f"[aggregate] Found {len(json_files)} result files.")

    rows = []
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)
        rows.append(data)

    if not rows:
        print("[aggregate] No results to aggregate. Exiting.")
        return

    df = pd.DataFrame(rows)
    if "status" in df.columns:
        ok_df = df[df["status"].fillna("ok") == "ok"].copy()
        failed_df = df[df["status"].fillna("ok") != "ok"].copy()
    else:
        ok_df = df.copy()
        failed_df = pd.DataFrame()

    expected = 45
    print(f"[aggregate] Completed configs: {len(ok_df)}/{expected}")
    if len(failed_df) > 0:
        print(f"[aggregate] Failed configs: {len(failed_df)}")

    if ok_df.empty:
        print("[aggregate] No successful runs to summarize. Exiting.")
        df.to_csv(out_dir / "gnn_ablation_summary.csv", index=False)
        return

    df = ok_df.sort_values("val_r2", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "gnn_ablation_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("  GNN ABLATION SWEEP RESULTS (sorted by val R²)")
    print(f"{'='*70}")
    cols_show = ["config", "top_k", "hidden_dim", "n_layers",
                 "val_r2", "val_mae", "n_params", "n_epochs_trained"]
    cols_present = [c for c in cols_show if c in df.columns]
    print(df[cols_present].to_string(index=False))

    # Best and worst
    best = df.iloc[0]
    worst = df.iloc[-1]
    print(f"\n  BEST:  {best['config']}  R²={best['val_r2']:.4f}")
    print(f"  WORST: {worst['config']}  R²={worst['val_r2']:.4f}")

    with open(out_dir / "gnn_ablation_best_config.json", "w") as f:
        json.dump(best.to_dict(), f, indent=2, default=str)

    # ── Heatmaps ───────────────────────────────────────────────────────
    if {"top_k", "hidden_dim", "n_layers", "val_r2"}.issubset(df.columns):
        # 1. Heatmap: top_k × hidden_dim (averaged over n_layers)
        pivot_kh = df.pivot_table(
            values="val_r2", index="top_k", columns="hidden_dim",
            aggfunc="mean",
        )
        # 2. Heatmap: top_k × n_layers (averaged over hidden_dim)
        pivot_kl = df.pivot_table(
            values="val_r2", index="top_k", columns="n_layers",
            aggfunc="mean",
        )
        # 3. Heatmap: hidden_dim × n_layers (averaged over top_k)
        pivot_hl = df.pivot_table(
            values="val_r2", index="hidden_dim", columns="n_layers",
            aggfunc="mean",
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        vmin = float(np.nanmin(df["val_r2"].values))
        vmax = float(np.nanmax(df["val_r2"].values))

        for ax, pivot, xlabel, ylabel, title in [
            (axes[0], pivot_kh, "hidden_dim", "top_k", "$R^2$ by ($k$, hidden dim)"),
            (axes[1], pivot_kl, "n_layers", "top_k", "$R^2$ by ($k$, depth)"),
            (axes[2], pivot_hl, "n_layers", "hidden_dim", "$R^2$ by (hidden dim, depth)"),
        ]:
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn",
                           origin="lower", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index)
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.set_title(title, fontsize=11)

            # Annotate cells
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                                fontsize=9, fontweight="bold",
                                color="white" if val < 0.03 else "black")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle("GAT committor ablation: validation $R^2$",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / "fig_gnn_ablation_heatmap.pdf",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[aggregate] Heatmap saved.")

    # ── Top vs bottom configuration summary plot ──────────────────────
    top_n = min(5, len(df))
    bottom_n = min(5, len(df))
    plot_df = pd.concat([df.head(top_n), df.tail(bottom_n)], axis=0).copy()
    plot_df["label"] = (
        plot_df["config"].astype(str)
        + "\n(k=" + plot_df["top_k"].astype(str)
        + ", h=" + plot_df["hidden_dim"].astype(str)
        + ", L=" + plot_df["n_layers"].astype(str) + ")"
    )
    colors = ["#4C72B0"] * top_n + ["#C44E52"] * bottom_n

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    ax.bar(range(len(plot_df)), plot_df["val_r2"].values, color=colors, alpha=0.85)
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Validation $R^2$", fontsize=11)
    ax.set_title("Best and worst GAT committor ablation configs", fontsize=12)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_gnn_ablation_best_vs_worst.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Key finding for thesis ─────────────────────────────────────────
    max_r2 = df["val_r2"].max()
    print(f"\n{'='*70}")
    print(f"  KEY FINDING FOR THESIS")
    print(f"{'='*70}")
    if max_r2 < 0.15:
        print(f"  Best R² across all 45 configs = {max_r2:.4f}")
        print(f"  → GNN failure is ROBUST to hyperparameter choice.")
        print(f"  → The committor is genuinely not recoverable via local")
        print(f"     message passing on these sparsified KTN graphs.")
    else:
        print(f"  Best R² across all 45 configs = {max_r2:.4f}")
        print(f"  → Some signal recoverable — report the best config.")

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
