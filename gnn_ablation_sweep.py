#!/usr/bin/env python
"""
gnn_ablation_sweep.py

Systematic hyperparameter ablation for GAT committor prediction.

Sweeps over:
    top_k     ∈ {5, 10, 20, 50, 100}
    hidden_dim ∈ {32, 64, 128}
    n_layers  ∈ {2, 3, 4}

Each configuration is identified by its SLURM_ARRAY_TASK_ID, so the full
sweep can be submitted as a single SLURM array job (45 configs total).

This script:
    1. Maps TASK_ID → (top_k, hidden_dim, n_layers)
    2. Loads the KTN dataset (cached from previous runs)
    3. Sparsifies to the given top_k
    4. Trains GAT for the committor task
    5. Saves metrics to {out_dir}/metrics_{config_name}.json

A companion script (gnn_ablation_aggregate.py) collects all JSON files
into a summary CSV and heatmap after all array tasks complete.

Usage (standalone — for testing a single config):
    python gnn_ablation_sweep.py --task-id 0

Usage (SLURM array — submit via run_gnn_ablation.sbatch):
    sbatch --array=0-44 run_gnn_ablation.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from ktn_dataset import KTNDataset
from train_gnn_v2 import sparsify_graph, enrich_node_features, train_single_config


# ======================================================================
#  Sweep grid
# ======================================================================

TOP_K_VALUES = [5, 10, 20, 50, 100]
HIDDEN_DIM_VALUES = [32, 64, 128]
N_LAYERS_VALUES = [2, 3, 4]

def build_grid():
    """Build a flat list of all (top_k, hidden_dim, n_layers) configs."""
    grid = []
    for k in TOP_K_VALUES:
        for h in HIDDEN_DIM_VALUES:
            for l in N_LAYERS_VALUES:
                grid.append({"top_k": k, "hidden_dim": h, "n_layers": l})
    return grid

GRID = build_grid()
N_CONFIGS = len(GRID)  # 5 × 3 × 3 = 45


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNN ablation sweep: single config run.")
    parser.add_argument("--task-id", type=int, required=True,
                        help="Config index (0 to 44). Set from SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--root", type=str, default="ktn_pyg_data_v2")
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--task", type=str, default="committor")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("gnn_ablation_results"))
    parser.add_argument("--overwrite", action="store_true",
                        help="Rerun even if metrics_{config}.json already exists.")
    args = parser.parse_args()

    if args.task_id < 0 or args.task_id >= N_CONFIGS:
        print(f"[ablation] task-id {args.task_id} out of range [0, {N_CONFIGS-1}]")
        sys.exit(1)

    config = GRID[args.task_id]
    top_k = config["top_k"]
    hidden_dim = config["hidden_dim"]
    n_layers = config["n_layers"]
    config_name = f"gat_k{top_k}_h{hidden_dim}_L{n_layers}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"metrics_{config_name}.json"

    if out_path.exists() and not args.overwrite:
        print(f"[ablation] Existing result found for {config_name}: {out_path}")
        print("[ablation] Skipping. Use --overwrite to rerun.")
        return

    print(f"\n{'='*60}")
    print(f"  GNN Ablation: config {args.task_id}/{N_CONFIGS-1}")
    print(f"  top_k={top_k}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"  config_name: {config_name}")
    print(f"{'='*60}\n")

    # ── Load dataset ───────────────────────────────────────────────────
    from config import BASE_DIR
    print("[ablation] Loading dataset...")
    dataset = KTNDataset(
        root=args.root,
        base_dir=BASE_DIR,
        T=300.0,
        targets_csv=args.targets_csv,
        compute_node_targets=True,
    )
    print(f"[ablation] {len(dataset)} graphs loaded.")

    # ── Sparsify ───────────────────────────────────────────────────────
    raw_list = list(dataset)
    print(f"[ablation] Sparsifying to top-{top_k}...")
    sparse_list = []
    for data in raw_list:
        sparse_list.append(sparsify_graph(data, top_k=top_k))

    # ── Train GAT ──────────────────────────────────────────────────────
    try:
        metrics = train_single_config(
            data_list=sparse_list,
            task=args.task,
            conv_type="gat",
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-4,
            patience=args.patience,
            batch_size=1,
            train_frac=0.8,
            seed=42,
            max_grad_norm=1.0,
            device=torch.device(args.device),
            config_name=config_name,
            out_dir=args.out_dir,
        )
    except Exception as exc:
        failure = {
            "config": config_name,
            "conv_type": "gat",
            "top_k": top_k,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "sweep_task_id": args.task_id,
            "status": "failed",
            "error": str(exc),
        }
        with open(out_path, "w") as f:
            json.dump(failure, f, indent=2)
        print(f"\n[ablation] FAILED for {config_name}: {exc}")
        print(f"[ablation] Failure record saved: {out_path}")
        return

    # ── Enrich metrics with sweep parameters ───────────────────────────
    if metrics:
        metrics.update({
            "conv_type": "gat",
            "top_k": top_k,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "sweep_task_id": args.task_id,
            "status": "ok",
        })
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[ablation] Saved: {out_path}")
    else:
        print(f"\n[ablation] No metrics returned for {config_name}.")


if __name__ == "__main__":
    main()
