#!/usr/bin/env python
"""
train_gnn_v2.py

Improved GNN node-level training with three enhancements over v1:
    1. Graph sparsification: keep only top-k edges per node by rate
    2. Rate-enriched node features: inject edge-weight statistics into nodes
    3. Automatic comparison of conv types (GCN, GAT, NNConv)

Designed for CPU-only training on dense KTN graphs.

Usage:
    python train_gnn_v2.py --top-k 20 --task committor
    python train_gnn_v2.py --top-k 20 --conv-types gcn gat nnconv
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ktn_dataset import KTNDataset
from gnn_models import KTNNodeModel


# ======================================================================
#  Enhancement 1: Graph sparsification
# ======================================================================

def sparsify_graph(data: Data, top_k: int = 20) -> Data:
    """
    Keep only the top-k strongest edges per node (by forward log-rate).

    For each node j, keep the top_k outgoing edges with highest transition
    rate. This preserves the kinetically dominant pathways while drastically
    reducing density.

    Also keeps all reverse edges of retained edges to maintain symmetry
    information.
    """
    edge_index = data.edge_index  # [2, E], (source, target)
    edge_attr = data.edge_attr    # [E, D_edge]
    N = data.x.shape[0]
    E = edge_index.shape[1]

    if E == 0 or top_k <= 0:
        return data

    src = edge_index[0].numpy()  # source nodes
    tgt = edge_index[1].numpy()  # target nodes

    # Use raw forward log-rate (col 0) for ranking.
    # Even after standardization, relative ordering is preserved.
    rates = edge_attr[:, 0].numpy()

    keep_mask = np.zeros(E, dtype=bool)

    # For each source node, keep its top-k outgoing edges
    for j in range(N):
        out_mask = src == j
        out_idx = np.where(out_mask)[0]
        if len(out_idx) <= top_k:
            keep_mask[out_idx] = True
        else:
            top_idx = out_idx[np.argsort(rates[out_idx])[-top_k:]]
            keep_mask[top_idx] = True

    # Also keep reverse edges of all retained edges to preserve symmetry info
    # Build a set of (src, tgt) pairs that are kept
    kept_pairs = set()
    kept_idx = np.where(keep_mask)[0]
    for idx in kept_idx:
        kept_pairs.add((src[idx], tgt[idx]))

    # Add reverse edges
    for idx in range(E):
        if not keep_mask[idx] and (tgt[idx], src[idx]) in kept_pairs:
            keep_mask[idx] = True

    # Apply mask
    new_data = data.clone()
    new_data.edge_index = edge_index[:, keep_mask]
    new_data.edge_attr = edge_attr[keep_mask]

    return new_data


# ======================================================================
#  Enhancement 2: Rate-enriched node features
# ======================================================================

def enrich_node_features(data: Data) -> Data:
    """
    Add rate-derived statistics to each node's feature vector.

    New features per node (appended to existing 9 features):
        9:  mean outgoing log-rate
        10: max outgoing log-rate
        11: mean incoming log-rate
        12: max incoming log-rate
        13: total branching probability (sum of outgoing B_ij)
        14: degree (number of edges, normalized by graph size)

    This gives GCN access to rate information through node features,
    even though GCN cannot use edge attributes directly.
    """
    x = data.x  # [N, D_node]
    edge_index = data.edge_index  # [2, E]
    edge_attr = data.edge_attr  # [E, D_edge]
    N = x.shape[0]

    # Initialize new features
    new_feats = torch.zeros(N, 6, dtype=torch.float32)

    if edge_index.shape[1] == 0:
        data = data.clone()
        data.x = torch.cat([x, new_feats], dim=1)
        return data

    src = edge_index[0]  # source nodes
    tgt = edge_index[1]  # target nodes
    fwd_log_rate = edge_attr[:, 0]  # col 0: forward log-rate
    branching = edge_attr[:, 2]     # col 2: branching probability

    for j in range(N):
        # Outgoing edges from node j
        out_mask = src == j
        if out_mask.any():
            out_rates = fwd_log_rate[out_mask]
            new_feats[j, 0] = out_rates.mean()
            new_feats[j, 1] = out_rates.max()
            new_feats[j, 4] = branching[out_mask].sum()

        # Incoming edges to node j
        in_mask = tgt == j
        if in_mask.any():
            in_rates = fwd_log_rate[in_mask]
            new_feats[j, 2] = in_rates.mean()
            new_feats[j, 3] = in_rates.max()

        # Normalized degree
        new_feats[j, 5] = float(out_mask.sum() + in_mask.sum()) / max(N, 1)

    data = data.clone()
    data.x = torch.cat([x, new_feats], dim=1)
    return data


# ======================================================================
#  Training loop
# ======================================================================

def train_single_config(
    data_list: list,
    task: str,
    conv_type: str,
    hidden_dim: int,
    n_layers: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    batch_size: int,
    train_frac: float,
    seed: int,
    max_grad_norm: float,
    device: torch.device,
    config_name: str,
    out_dir: Path,
) -> Dict:
    """Train a single node-level model configuration and return metrics."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    sample = data_list[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    # For GCN/GIN, edge_dim is not used but we still pass 0
    model = KTNNodeModel(
        node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, conv_type=conv_type, task=task,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  [{config_name}] node_dim={node_dim}, edge_dim={edge_dim}, "
          f"params={n_params:,}, conv={conv_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5,
    )

    # Build train/val masks
    target_attr = "committor" if task == "committor" else "mfpt_to_B"
    ready_list = []
    for graph_idx, data in enumerate(data_list):
        if not hasattr(data, target_attr) or getattr(data, target_attr) is None:
            continue

        N = data.x.shape[0]
        interior = ~(data.A_mask | data.B_mask)
        interior_idx = torch.where(interior)[0].numpy()

        rng = np.random.default_rng(seed + 1009 * graph_idx)
        rng.shuffle(interior_idx)
        if len(interior_idx) <= 1:
            split = len(interior_idx)
        else:
            split = int(len(interior_idx) * train_frac)
            split = min(max(split, 1), len(interior_idx) - 1)

        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[data.A_mask] = True
        train_mask[data.B_mask] = True
        train_mask[torch.from_numpy(interior_idx[:split])] = True
        val_mask[torch.from_numpy(interior_idx[split:])] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.node_target = getattr(data, target_attr)
        ready_list.append(data)

    if not ready_list:
        print(f"  [{config_name}] No valid graphs. Skipping.")
        return {}

    n_val_total = sum(int(d.val_mask.sum()) for d in ready_list)
    if n_val_total == 0:
        print(f"  [{config_name}] No validation nodes. Skipping.")
        return {}

    loader = DataLoader(ready_list, batch_size=batch_size, shuffle=True)

    if task == "committor":
        loss_fn = nn.BCELoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}
    wait = 0

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0
        n_train = 0

        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.node_target
            losses = loss_fn(pred, target)
            train_loss = losses[batch.train_mask].mean()
            optimizer.zero_grad()
            train_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            total_train_loss += train_loss.item() * batch.train_mask.sum().item()
            n_train += batch.train_mask.sum().item()

        avg_train = total_train_loss / max(n_train, 1)

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.node_target
                losses = loss_fn(pred, target)
                total_val_loss += losses[batch.val_mask].sum().item()
                n_val += batch.val_mask.sum().item()

        avg_val = total_val_loss / max(n_val, 1)
        scheduler.step(avg_val)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{config_name}] Early stop epoch {epoch+1} "
                      f"(best_val={best_val_loss:.6f})")
                break

        if (epoch + 1) % 25 == 0:
            print(f"  [{config_name}] Epoch {epoch+1:4d}  "
                  f"train={avg_train:.6f}  val={avg_val:.6f}")

    # Final evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in DataLoader(ready_list, batch_size=1):
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            target = batch.node_target.cpu().numpy()
            val_mask = batch.val_mask.cpu().numpy()
            if np.any(val_mask):
                all_pred.append(pred[val_mask])
                all_true.append(target[val_mask])

    if not all_pred:
        return {}

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    from sklearn.metrics import r2_score, mean_absolute_error
    val_r2 = float(r2_score(all_true, all_pred)) if all_pred.size >= 2 else float("nan")
    val_mae = float(mean_absolute_error(all_true, all_pred))

    metrics = {
        "config": config_name,
        "conv_type": conv_type,
        "task": task,
        "val_r2": val_r2,
        "val_mae": val_mae,
        "n_val_nodes": int(all_pred.size),
        "best_val_loss": float(best_val_loss),
        "n_params": n_params,
        "n_epochs_trained": len(history["train_loss"]),
    }

    print(f"  [{config_name}] R² = {val_r2:.4f}, MAE = {val_mae:.6f}")

    # Save model and metrics
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"model_{config_name}.pt")
    with open(out_dir / f"metrics_{config_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{config_name}: Training Curves")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].scatter(all_true, all_pred, s=1, alpha=0.1)
    lo, hi = all_true.min(), all_true.max()
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_xlabel(f"True {task}")
    axes[1].set_ylabel(f"Predicted {task}")
    axes[1].set_title(f"R² = {val_r2:.4f}")

    fig.tight_layout()
    fig.savefig(out_dir / f"results_{config_name}.png", dpi=200)
    plt.close(fig)

    return metrics


# ======================================================================
#  Main: systematic comparison
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNN v2: sparsification + rate-enriched features + conv comparison")
    parser.add_argument("--root", type=str, default="ktn_pyg_data")
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--task", type=str, default="committor",
                        choices=["committor", "mfpt"])
    parser.add_argument("--top-k", type=int, default=20,
                        help="Keep top-k edges per node. 0 = no sparsification.")
    parser.add_argument("--conv-types", nargs="+", default=["gcn", "gat", "nnconv"],
                        help="Conv types to compare.")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("gnn_results_v2"))
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load dataset
    from config import BASE_DIR
    print("[gnn_v2] Loading dataset...")
    dataset = KTNDataset(
        root=args.root,
        base_dir=BASE_DIR,
        T=300.0,
        targets_csv=args.targets_csv,
        compute_node_targets=True,
    )
    print(f"[gnn_v2] {len(dataset)} graphs loaded.")

    # Prepare data variants
    raw_list = list(dataset)

    # Step 1: Sparsify if requested
    if args.top_k > 0:
        print(f"\n[gnn_v2] Sparsifying to top-{args.top_k} edges per node...")
        sparse_list = []
        for i, data in enumerate(raw_list):
            orig_edges = data.edge_index.shape[1]
            sp = sparsify_graph(data, top_k=args.top_k)
            new_edges = sp.edge_index.shape[1]
            seq = getattr(data, "sequence", f"g{i}")
            print(f"  {seq}: {orig_edges:,} -> {new_edges:,} edges "
                  f"({100*new_edges/max(orig_edges,1):.1f}%)")
            sparse_list.append(sp)
    else:
        sparse_list = raw_list

    # Step 2: Create enriched-feature version
    print(f"\n[gnn_v2] Enriching node features with rate statistics...")
    enriched_list = [enrich_node_features(d) for d in sparse_list]

    print(f"[gnn_v2] Node features: {raw_list[0].x.shape[1]} (original) -> "
          f"{enriched_list[0].x.shape[1]} (enriched)")

    # Step 3: Run experiments
    all_results = []

    for conv_type in args.conv_types:
        # Decide which data variant to use:
        # - GCN/GIN: use enriched features (they can't use edge attrs)
        # - GAT/NNConv: use sparse data (they use edge attrs directly)
        if conv_type in ("gcn", "gin"):
            data = enriched_list
            suffix = f"{conv_type}_enriched_k{args.top_k}"
        else:
            data = sparse_list
            suffix = f"{conv_type}_sparse_k{args.top_k}"

        print(f"\n{'='*60}")
        print(f"  Experiment: {suffix}")
        print(f"{'='*60}")

        metrics = train_single_config(
            data_list=data,
            task=args.task,
            conv_type=conv_type,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_epochs=args.epochs,
            lr=args.lr,
            weight_decay=1e-4,
            patience=args.patience,
            batch_size=args.batch_size,
            train_frac=0.8,
            seed=42,
            max_grad_norm=1.0,
            device=device,
            config_name=suffix,
            out_dir=args.out_dir,
        )
        if metrics:
            all_results.append(metrics)

    # Summary comparison
    if all_results:
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY ({args.task})")
        print(f"{'='*60}")
        print(f"  {'Config':<35s} {'R²':>8s} {'MAE':>10s} {'Params':>10s}")
        print(f"  {'-'*35} {'-'*8} {'-'*10} {'-'*10}")

        # Sort by R² descending
        all_results.sort(key=lambda m: m.get("val_r2", -999), reverse=True)
        for m in all_results:
            print(f"  {m['config']:<35s} {m['val_r2']:>8.4f} "
                  f"{m['val_mae']:>10.6f} {m['n_params']:>10,}")

        # Save summary
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with open(args.out_dir / "comparison_summary.json", "w") as f:
            json.dump(all_results, f, indent=2)

        # Comparison bar plot
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [m["config"] for m in all_results]
        r2s = [m["val_r2"] for m in all_results]
        colors = ["#2ecc71" if r > 0.1 else "#e74c3c" if r < 0 else "#f39c12"
                  for r in r2s]
        bars = ax.barh(range(len(names)), r2s, color=colors, edgecolor="k", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Validation R²")
        ax.set_title(f"GNN v2 Comparison: {args.task} (top-{args.top_k} sparsification)")
        ax.axvline(x=0, color="k", linewidth=0.5)
        for i, (bar, r2) in enumerate(zip(bars, r2s)):
            ax.text(max(r2, 0) + 0.01, i, f"{r2:.4f}", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / "comparison.png", dpi=200)
        plt.close(fig)

    print(f"\n[gnn_v2] Done. Results in {args.out_dir}")


if __name__ == "__main__":
    main()
