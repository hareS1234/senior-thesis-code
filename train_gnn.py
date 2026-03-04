#!/usr/bin/env python
"""
train_gnn.py

Training and evaluation for GNN models on KTN data.

Supports three modes:
    --mode node     : Node-level training (committor or MFPT)
    --mode graph    : Graph-level LOO-CV (predict MFPT_AB, t1, etc.)
    --mode multitask: Pretrain on node targets, finetune for graph targets

Usage:
    python train_gnn.py --mode node   --root ktn_pyg_data --task committor
    python train_gnn.py --mode graph  --root ktn_pyg_data --target 0
    python train_gnn.py --mode multitask --root ktn_pyg_data
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

from torch_geometric.loader import DataLoader

from ktn_dataset import KTNDataset
from gnn_models import KTNNodeModel, KTNGraphModel, KTNMultiTaskModel

# Graph-level target columns (indices into data.y[0])
TARGET_NAMES = ["log_MFPT_coarse_AB", "log_MFPT_coarse_BA", "log_t1", "t1_over_t2"]


# ======================================================================
#  Node-level training
# ======================================================================

def train_node_level(
    dataset,
    task: str = "committor",
    hidden_dim: int = 64,
    n_layers: int = 3,
    conv_type: str = "nnconv",
    n_epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    train_frac: float = 0.8,
    seed: int = 42,
    max_grad_norm: float = 1.0,
    out_dir: Path = Path("gnn_results"),
) -> Dict[str, float]:
    """
    Train node-level model across all graphs.

    80% of interior nodes (not A/B) are training, 20% validation.
    A/B nodes have fixed targets and are always in training.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine feature dims from first graph
    sample = dataset[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0

    model = KTNNodeModel(
        node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
        n_layers=n_layers, conv_type=conv_type, task=task,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5,
    )

    # Build train/val masks for each graph
    data_list = []
    target_attr = "committor" if task == "committor" else "mfpt_to_B"
    for graph_idx, data in enumerate(dataset):
        if not hasattr(data, target_attr):
            continue

        target = getattr(data, target_attr)
        if target is None:
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

        # A/B always train
        train_mask[data.A_mask] = True
        train_mask[data.B_mask] = True
        train_mask[torch.from_numpy(interior_idx[:split])] = True
        val_mask[torch.from_numpy(interior_idx[split:])] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.node_target = target
        data_list.append(data)

    if len(data_list) == 0:
        print("[train_gnn] No graphs with valid node targets.")
        return {}

    n_val_total = int(sum(int(d.val_mask.sum().item()) for d in data_list))
    if n_val_total == 0:
        print("[train_gnn] No validation nodes available after splitting. Skipping.")
        return {}

    loader = DataLoader(data_list, batch_size=4, shuffle=True)

    # Loss function
    if task == "committor":
        loss_fn = nn.BCELoss(reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": []}

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

        avg_train_loss = total_train_loss / max(n_train, 1)

        # Validation
        model.eval()
        total_val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                pred = model(batch)
                target = batch.node_target
                losses = loss_fn(pred, target)
                val_loss_batch = losses[batch.val_mask].sum().item()
                total_val_loss += val_loss_batch
                n_val += batch.val_mask.sum().item()

        avg_val_loss = total_val_loss / max(n_val, 1)
        scheduler.step(avg_val_loss)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:4d}  train_loss={avg_train_loss:.6f}  "
                  f"val_loss={avg_val_loss:.6f}")

    # Final evaluation with best model
    model.load_state_dict(best_state)
    model.eval()

    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in DataLoader(data_list, batch_size=1):
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            target = batch.node_target.cpu().numpy()
            val_mask = batch.val_mask.cpu().numpy()
            if np.any(val_mask):
                all_pred.append(pred[val_mask])
                all_true.append(target[val_mask])

    if not all_pred:
        print("[train_gnn] Validation split produced no evaluable nodes. Skipping.")
        return {}

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    from sklearn.metrics import r2_score, mean_absolute_error
    metrics = {
        "task": task,
        "val_r2": float(r2_score(all_true, all_pred)),
        "val_mae": float(mean_absolute_error(all_true, all_pred)),
        "n_val_nodes": int(all_pred.size),
        "best_val_loss": float(best_val_loss),
    }
    print(f"\n  Node-level {task}: val R² = {metrics['val_r2']:.4f}, "
          f"val MAE = {metrics['val_mae']:.6f}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / f"node_model_{task}.pt")

    with open(out_dir / f"node_metrics_{task}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Node-level {task}: Training Curves")
    axes[0].legend()
    axes[0].set_yscale("log")

    axes[1].scatter(all_true, all_pred, s=1, alpha=0.1)
    lo, hi = all_true.min(), all_true.max()
    axes[1].plot([lo, hi], [lo, hi], "r--", linewidth=1)
    axes[1].set_xlabel(f"True {task}")
    axes[1].set_ylabel(f"Predicted {task}")
    axes[1].set_title(f"R² = {metrics['val_r2']:.4f}")

    fig.tight_layout()
    fig.savefig(out_dir / f"node_{task}_results.png", dpi=200)
    plt.close(fig)

    return metrics


# ======================================================================
#  Graph-level LOO-CV
# ======================================================================

def train_graph_level_loocv(
    dataset,
    target_idx: int = 0,
    hidden_dim: int = 64,
    n_layers: int = 3,
    conv_type: str = "nnconv",
    n_epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    n_seeds: int = 5,
    max_grad_norm: float = 1.0,
    out_dir: Path = Path("gnn_results"),
) -> Dict[str, float]:
    """
    Leave-one-out CV for graph-level prediction.

    For each held-out graph: train fresh model on N-1 graphs, predict held-out.
    Ensemble over n_seeds random initializations.
    """
    target_name = TARGET_NAMES[target_idx]

    # Filter to graphs with valid targets
    valid_data = []
    for data in dataset:
        if data.y is not None and not torch.isnan(data.y[0, target_idx]):
            valid_data.append(data)

    N = len(valid_data)
    if N < 5:
        print(f"[train_gnn] Only {N} graphs with valid {target_name}. Skipping.")
        return {}

    print(f"\n[train_gnn] Graph-level LOO-CV for {target_name} (N={N}, seeds={n_seeds})")

    sample = valid_data[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true = np.array([d.y[0, target_idx].item() for d in valid_data])
    y_pred_all = np.zeros((N, n_seeds))

    for fold_idx in range(N):
        train_data = [valid_data[i] for i in range(N) if i != fold_idx]
        test_data = valid_data[fold_idx]
        seq_name = getattr(test_data, "sequence", f"graph_{fold_idx}")

        for seed in range(n_seeds):
            torch.manual_seed(seed * 1000 + fold_idx)

            model = KTNGraphModel(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, conv_type=conv_type, n_targets=1,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)

            # Split training into train/val for early stopping
            n_train_graphs = len(train_data)
            n_val = max(1, n_train_graphs // 6)
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_train_graphs)
            val_indices = perm[:n_val]
            train_indices = perm[n_val:]

            train_subset = [train_data[i] for i in train_indices]
            val_subset = [train_data[i] for i in val_indices]

            train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=len(val_subset))

            best_val = float("inf")
            best_state = copy.deepcopy(model.state_dict())
            wait = 0

            for epoch in range(n_epochs):
                model.train()
                for batch in train_loader:
                    batch = batch.to(device)
                    pred = model(batch).squeeze(-1)
                    target = batch.y[:, target_idx]
                    loss = F.mse_loss(pred, target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                # Validation
                model.eval()
                val_loss = float("inf")
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        pred = model(batch).squeeze(-1)
                        target = batch.y[:, target_idx]
                        val_loss = F.mse_loss(pred, target).item()

                if val_loss < best_val:
                    best_val = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            # Predict held-out graph
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(DataLoader([test_data], batch_size=1)))
                test_batch = test_batch.to(device)
                pred = model(test_batch).squeeze().item()

            y_pred_all[fold_idx, seed] = pred

        # Average over seeds
        y_pred_avg = y_pred_all[fold_idx].mean()
        print(f"  Fold {fold_idx+1:2d}/{N} ({seq_name:15s}): "
              f"true={y_true[fold_idx]:.4f}, pred={y_pred_avg:.4f}")

    # Ensemble average predictions
    y_pred_final = y_pred_all.mean(axis=1)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    metrics = {
        "target": target_name,
        "n_graphs": N,
        "n_seeds": n_seeds,
        "r2": float(r2_score(y_true, y_pred_final)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_final))),
        "mae": float(mean_absolute_error(y_true, y_pred_final)),
    }
    print(f"\n  {target_name} LOO-CV: R² = {metrics['r2']:.3f}, "
          f"RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"graph_loocv_{target_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(out_dir / f"graph_loocv_{target_name}_preds.npz",
             y_true=y_true, y_pred=y_pred_final,
             y_pred_all_seeds=y_pred_all)

    # Plot
    labels = [getattr(d, "sequence", f"g{i}") for i, d in enumerate(valid_data)]
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    ax.scatter(y_true, y_pred_final, s=50, alpha=0.7, edgecolors="k", linewidths=0.5)
    for i, (xt, xp) in enumerate(zip(y_true, y_pred_final)):
        ax.annotate(labels[i], (xt, xp), fontsize=6, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")
    lo, hi = min(y_true.min(), y_pred_final.min()), max(y_true.max(), y_pred_final.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.3)
    ax.set_xlabel(f"Actual {target_name}")
    ax.set_ylabel(f"Predicted {target_name}")
    ax.set_title(f"GNN LOO-CV: {target_name} (R² = {metrics['r2']:.3f})")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"graph_loocv_{target_name}.png", dpi=200)
    plt.close(fig)

    return metrics


# ======================================================================
#  Multi-task: pretrain node → finetune graph
# ======================================================================

def train_multitask(
    dataset,
    target_idx: int = 0,
    node_task: str = "committor",
    hidden_dim: int = 64,
    n_layers: int = 3,
    conv_type: str = "nnconv",
    pretrain_epochs: int = 100,
    finetune_epochs: int = 200,
    alpha: float = 0.5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 50,
    n_seeds: int = 5,
    max_grad_norm: float = 1.0,
    out_dir: Path = Path("gnn_results"),
) -> Dict[str, float]:
    """
    Two-stage training with LOO-CV:
        1. Pre-train shared backbone on node-level task (all training graphs)
        2. Fine-tune graph head (freeze backbone optionally)
    """
    target_name = TARGET_NAMES[target_idx]
    target_attr = "committor" if node_task == "committor" else "mfpt_to_B"

    # Filter to graphs with both node and graph targets
    valid_data = []
    for data in dataset:
        has_graph = data.y is not None and not torch.isnan(data.y[0, target_idx])
        has_node = hasattr(data, target_attr) and getattr(data, target_attr) is not None
        if has_graph and has_node:
            valid_data.append(data)

    N = len(valid_data)
    if N < 5:
        print(f"[train_gnn] Only {N} graphs with both targets. Skipping multitask.")
        return {}

    print(f"\n[train_gnn] Multi-task LOO-CV: node={node_task}, "
          f"graph={target_name} (N={N})")

    sample = valid_data[0]
    node_dim = sample.x.shape[1]
    edge_dim = sample.edge_attr.shape[1] if sample.edge_attr is not None else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_true = np.array([d.y[0, target_idx].item() for d in valid_data])
    y_pred_all = np.zeros((N, n_seeds))

    if node_task == "committor":
        node_loss_fn = nn.BCELoss()
    else:
        node_loss_fn = nn.MSELoss()

    for fold_idx in range(N):
        train_data = [valid_data[i] for i in range(N) if i != fold_idx]
        test_data = valid_data[fold_idx]

        for seed in range(n_seeds):
            torch.manual_seed(seed * 1000 + fold_idx)

            model = KTNMultiTaskModel(
                node_dim=node_dim, edge_dim=edge_dim, hidden_dim=hidden_dim,
                n_layers=n_layers, conv_type=conv_type,
                n_graph_targets=1, node_task=node_task,
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         weight_decay=weight_decay)

            train_loader = DataLoader(train_data, batch_size=4, shuffle=True)

            # Stage 1: Joint pre-training with early stopping
            best_pretrain_loss = float("inf")
            best_pretrain_state = copy.deepcopy(model.state_dict())
            pretrain_wait = 0

            for epoch in range(pretrain_epochs):
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    node_pred, graph_pred = model(batch)

                    # Node loss (on all nodes)
                    node_target = getattr(batch, target_attr)
                    n_loss = node_loss_fn(node_pred, node_target)

                    # Graph loss
                    g_target = batch.y[:, target_idx]
                    g_loss = F.mse_loss(graph_pred.squeeze(-1), g_target)

                    loss = alpha * n_loss + (1 - alpha) * g_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if avg_loss < best_pretrain_loss:
                    best_pretrain_loss = avg_loss
                    best_pretrain_state = copy.deepcopy(model.state_dict())
                    pretrain_wait = 0
                else:
                    pretrain_wait += 1
                    if pretrain_wait >= patience:
                        break

            model.load_state_dict(best_pretrain_state)

            # Stage 2: Fine-tune graph head only
            for param in model.backbone.parameters():
                param.requires_grad = False
            optimizer = torch.optim.Adam(
                model.graph_head.parameters(), lr=lr * 0.1, weight_decay=weight_decay,
            )

            best_loss = float("inf")
            best_state = copy.deepcopy(model.state_dict())
            finetune_wait = 0

            for epoch in range(finetune_epochs):
                model.train()
                epoch_loss = 0.0
                n_batches = 0
                for batch in train_loader:
                    batch = batch.to(device)
                    _, graph_pred = model(batch)
                    g_target = batch.y[:, target_idx]
                    loss = F.mse_loss(graph_pred.squeeze(-1), g_target)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.graph_head.parameters(), max_grad_norm)
                    optimizer.step()
                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_state = copy.deepcopy(model.state_dict())
                    finetune_wait = 0
                else:
                    finetune_wait += 1
                    if finetune_wait >= patience:
                        break

            # Predict held-out
            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(DataLoader([test_data], batch_size=1)))
                test_batch = test_batch.to(device)
                _, pred = model(test_batch)
                y_pred_all[fold_idx, seed] = pred.squeeze().item()

        seq_name = getattr(test_data, "sequence", f"graph_{fold_idx}")
        print(f"  Fold {fold_idx+1:2d}/{N} ({seq_name:15s}): "
              f"true={y_true[fold_idx]:.4f}, "
              f"pred={y_pred_all[fold_idx].mean():.4f}")

    y_pred_final = y_pred_all.mean(axis=1)

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    metrics = {
        "target": target_name,
        "node_task": node_task,
        "n_graphs": N,
        "r2": float(r2_score(y_true, y_pred_final)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_final))),
        "mae": float(mean_absolute_error(y_true, y_pred_final)),
    }
    print(f"\n  Multitask {target_name}: R² = {metrics['r2']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"multitask_{target_name}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez(out_dir / f"multitask_{target_name}_preds.npz",
             y_true=y_true, y_pred=y_pred_final)

    return metrics


# ======================================================================
#  Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train GNN on KTN data.")
    parser.add_argument("--mode", choices=["node", "graph", "multitask", "all"],
                        default="all", help="Training mode.")
    parser.add_argument("--root", type=str, default="ktn_pyg_data",
                        help="PyG dataset root directory.")
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--task", type=str, default="committor",
                        choices=["committor", "mfpt"],
                        help="Node-level task.")
    parser.add_argument("--target", type=int, default=0,
                        help=f"Graph-level target index: {dict(enumerate(TARGET_NAMES))}")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--conv-type", type=str, default="nnconv",
                        choices=["nnconv", "gat", "gcn", "gin"])
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("gnn_results"))
    args = parser.parse_args()

    from config import BASE_DIR
    print("[train_gnn] Loading dataset...")
    dataset = KTNDataset(
        root=args.root,
        base_dir=BASE_DIR,
        T=300.0,
        targets_csv=args.targets_csv,
    )
    print(f"[train_gnn] {len(dataset)} graphs loaded.")

    if args.mode in ("node", "all"):
        print("\n" + "=" * 60)
        print("  NODE-LEVEL TRAINING")
        print("=" * 60)
        train_node_level(
            dataset, task=args.task,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
            conv_type=args.conv_type, out_dir=args.out_dir,
        )

    if args.mode in ("graph", "all"):
        print("\n" + "=" * 60)
        print("  GRAPH-LEVEL LOO-CV")
        print("=" * 60)
        for t_idx in range(len(TARGET_NAMES)):
            if args.mode == "graph" and t_idx != args.target:
                continue
            train_graph_level_loocv(
                dataset, target_idx=t_idx,
                hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                conv_type=args.conv_type, n_seeds=args.n_seeds,
                out_dir=args.out_dir,
            )

    if args.mode in ("multitask", "all"):
        print("\n" + "=" * 60)
        print("  MULTI-TASK (PRETRAIN NODE → FINETUNE GRAPH)")
        print("=" * 60)
        for t_idx in range(len(TARGET_NAMES)):
            train_multitask(
                dataset, target_idx=t_idx, node_task=args.task,
                hidden_dim=args.hidden_dim, n_layers=args.n_layers,
                conv_type=args.conv_type, n_seeds=args.n_seeds,
                out_dir=args.out_dir,
            )

    print("\n[train_gnn] Done. Results in", args.out_dir)


if __name__ == "__main__":
    main()
