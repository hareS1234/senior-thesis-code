#!/usr/bin/env python
"""
ktn_dataset.py

Convert coarse-grained KTN data (sparse matrices, node/edge features) into
PyTorch Geometric Data objects for GNN training.

Each KTN becomes a Data object with:
    x           : node features  [N, D_node]
    edge_index  : COO connectivity  [2, E]
    edge_attr   : edge features  [E, D_edge]
    y           : graph-level target(s)  [1, n_targets]
    committor   : node-level committor q_i  [N]  (optional)
    mfpt_to_B   : node-level MFPT to B  [N]  (optional)

Usage:
    dataset = KTNDataset(
        root="ktn_pyg_data",
        base_dir=Path("/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped"),
        targets_csv=Path("GTcheck_micro_vs_coarse_T300K_full.csv"),
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

import torch
from torch_geometric.data import Data, InMemoryDataset

from config import iter_dps_dirs
from io_markov import load_markov, load_AB_selectors, temp_tag


# ======================================================================
#  Node-level target computation
# ======================================================================

def compute_committor(
    Q: csr_matrix,
    A_sel: np.ndarray,
    B_sel: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute the forward committor q_i^+ for each node.

    q_i^+ = P(reach B before A | start at i).
    Boundary: q_A = 0, q_B = 1.
    Interior: Q_II @ q_I = -Q_IB @ 1_B

    Returns (N,) array or None on failure.
    """
    N = Q.shape[0]
    A_idx = np.where(A_sel)[0]
    B_idx = np.where(B_sel)[0]
    I_mask = ~(A_sel | B_sel)
    I_idx = np.where(I_mask)[0]

    if I_idx.size == 0:
        # Only A and B states, no interior
        q = np.zeros(N)
        q[B_sel] = 1.0
        return q

    # Extract submatrix Q_II and Q_IB
    Q_II = Q[np.ix_(I_idx, I_idx)]
    Q_IB = Q[np.ix_(I_idx, B_idx)]

    # Solve Q_II @ q_I = -Q_IB @ ones
    rhs = -Q_IB @ np.ones(B_idx.size)

    try:
        q_I = spsolve(Q_II, rhs)
    except Exception:
        return None

    # Clamp to [0, 1] (numerical noise)
    q_I = np.clip(q_I, 0.0, 1.0)

    q = np.zeros(N)
    q[B_sel] = 1.0
    q[I_idx] = q_I
    return q


def compute_mfpt_to_B(
    Q: csr_matrix,
    B_sel: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute MFPT from each node to set B.

    Interior: Q_II @ m_I = -1  (with m_B = 0).

    Returns (N,) array or None on failure.
    """
    N = Q.shape[0]
    B_idx = np.where(B_sel)[0]
    I_mask = ~B_sel
    I_idx = np.where(I_mask)[0]

    if I_idx.size == 0:
        return np.zeros(N)

    Q_II = Q[np.ix_(I_idx, I_idx)]
    rhs = -np.ones(I_idx.size)

    try:
        m_I = spsolve(Q_II, rhs)
    except Exception:
        return None

    m = np.zeros(N)
    m[I_idx] = np.clip(m_I, 0.0, None)
    return m


# ======================================================================
#  Feature building
# ======================================================================

def build_node_features(
    pi: np.ndarray,
    tau: np.ndarray,
    A_sel: np.ndarray,
    B_sel: np.ndarray,
    energies: Optional[np.ndarray] = None,
    entropies: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Build node feature matrix [N, D_node].

    Features (standardized within each graph):
        0: log(pi)
        1: log(tau)
        2: energy        (if available, else 0)
        3: entropy       (if available, else 0)
        4: is_A          (binary)
        5: is_B          (binary)
    """
    N = pi.size
    feats = np.zeros((N, 6), dtype=np.float32)

    feats[:, 0] = np.log(np.clip(pi, 1e-300, None))
    feats[:, 1] = np.log(np.clip(tau, 1e-300, None))

    if energies is not None and energies.size == N:
        feats[:, 2] = energies
    if entropies is not None and entropies.size == N:
        feats[:, 3] = entropies

    # Standardize continuous features (columns 0-3) within graph
    for col in range(4):
        vals = feats[:, col]
        finite = vals[np.isfinite(vals)]
        if finite.size > 1 and finite.std() > 0:
            feats[:, col] = (vals - finite.mean()) / finite.std()
        else:
            feats[:, col] = 0.0

    feats[:, 4] = A_sel.astype(np.float32)
    feats[:, 5] = B_sel.astype(np.float32)

    return torch.from_numpy(feats)


def build_edge_features(
    K: csr_matrix,
    B_mat: csr_matrix,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index [2, E] and edge_attr [E, D_edge] from rate matrix.

    Edge features:
        0: log(k_ij)     -- forward rate (K[i,j] = rate j->i)
        1: log(k_ji)     -- reverse rate
        2: b_ij          -- branching probability

    K convention: K[i,j] = rate of transition *into* i *from* j.
    So edge (source=j, target=i) has rate K[i,j].
    """
    K_coo = K.tocoo()
    # Only off-diagonal entries
    mask = K_coo.row != K_coo.col
    rows = K_coo.row[mask]  # target
    cols = K_coo.col[mask]  # source
    rates = K_coo.data[mask]

    # Edge index: source -> target convention for PyG
    edge_index = torch.tensor(np.vstack([cols, rows]), dtype=torch.long)

    # Edge features
    n_edges = cols.size
    edge_attr = np.zeros((n_edges, 3), dtype=np.float32)

    # Forward rate: log(k_{i<-j}) = log(K[i,j])
    edge_attr[:, 0] = np.log(np.clip(rates, 1e-300, None))

    # Reverse rate: log(k_{j<-i}) = log(K[j,i])
    K_csr = K.tocsr()
    for idx in range(n_edges):
        rev_rate = K_csr[cols[idx], rows[idx]]
        edge_attr[idx, 1] = np.log(max(float(rev_rate), 1e-300))

    # Branching probability
    B_coo = B_mat.tocoo()
    B_dict = {}
    for r, c, d in zip(B_coo.row, B_coo.col, B_coo.data):
        B_dict[(r, c)] = d
    for idx in range(n_edges):
        edge_attr[idx, 2] = B_dict.get((rows[idx], cols[idx]), 0.0)

    # Standardize log-rate features
    for col in range(2):
        vals = edge_attr[:, col]
        finite = vals[np.isfinite(vals)]
        if finite.size > 1 and finite.std() > 0:
            edge_attr[:, col] = (vals - finite.mean()) / finite.std()

    return edge_index, torch.from_numpy(edge_attr)


# ======================================================================
#  Dataset
# ======================================================================

class KTNDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for hexapeptide KTNs.

    Args:
        root: directory for processed .pt cache
        base_dir: path to LAMMPS_uncapped (where DPS dirs live)
        T: temperature (default 300.0)
        targets_csv: CSV with graph-level targets (from GTcheck)
        compute_node_targets: whether to solve for committor/MFPT (slower)
    """

    def __init__(
        self,
        root: str,
        base_dir: Optional[Path] = None,
        T: float = 300.0,
        targets_csv: Optional[Path] = None,
        compute_node_targets: bool = True,
        transform=None,
        pre_transform=None,
    ):
        self.base_dir = base_dir
        self.T = T
        self.targets_csv = targets_csv
        self.compute_node_targets = compute_node_targets
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ["ktn_data.pt"]

    def process(self):
        # Load graph-level targets
        graph_targets = {}
        if self.targets_csv is not None and self.targets_csv.exists():
            tgt_df = pd.read_csv(self.targets_csv)
            tgt_df["dps_dir"] = tgt_df["dps_dir"].astype(str).str.rstrip("/")
            for _, row in tgt_df.iterrows():
                dps = row["dps_dir"]
                targets = {}
                for col in ["MFPT_coarse_AB", "MFPT_coarse_BA", "t1", "t1_over_t2"]:
                    if col in row and pd.notna(row[col]):
                        val = float(row[col])
                        targets[col] = val
                        if val > 0:
                            targets[f"log_{col}"] = np.log10(val)
                graph_targets[dps] = targets

        dps_dirs = iter_dps_dirs()
        tag = temp_tag(self.T)
        data_list = []

        for dps_dir in dps_dirs:
            name = dps_dir.name
            coarse_dir = dps_dir / f"markov_{tag}" / f"GT_kept_{tag}"
            if not coarse_dir.exists():
                continue

            try:
                B_mat, K, Q, tau, pi = load_markov(dps_dir, self.T, coarse=True)
            except Exception as e:
                print(f"  [ktn_dataset] Skipping {name}: {e}")
                continue

            A_sel, B_sel = load_AB_selectors(dps_dir, self.T, coarse=True)
            if A_sel is None or B_sel is None:
                continue
            if A_sel.sum() == 0 or B_sel.sum() == 0:
                continue

            # Try loading energies/entropies for node features
            eff_dir = coarse_dir
            energies, entropies = None, None
            epath = eff_dir / f"energies_eff_{tag}.npy"
            spath = eff_dir / f"entropies_eff_{tag}.npy"
            if epath.exists():
                energies = np.load(epath)
            if spath.exists():
                entropies = np.load(spath)

            # Build features
            x = build_node_features(pi, tau, A_sel, B_sel, energies, entropies)
            edge_index, edge_attr = build_edge_features(K, B_mat)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

            # Graph-level targets
            dps_key = str(dps_dir).rstrip("/")
            if dps_key in graph_targets:
                gt = graph_targets[dps_key]
                y_vals = [
                    gt.get("log_MFPT_coarse_AB", float("nan")),
                    gt.get("log_MFPT_coarse_BA", float("nan")),
                    gt.get("log_t1", float("nan")),
                    gt.get("t1_over_t2", float("nan")),
                ]
                data.y = torch.tensor([y_vals], dtype=torch.float)
            else:
                data.y = torch.full((1, 4), float("nan"))

            # Node-level targets
            if self.compute_node_targets:
                committor = compute_committor(Q, A_sel, B_sel)
                if committor is not None:
                    data.committor = torch.from_numpy(committor.astype(np.float32))

                mfpt = compute_mfpt_to_B(Q, B_sel)
                if mfpt is not None:
                    # Log-transform for numerical stability
                    mfpt_log = np.log10(np.clip(mfpt, 1e-300, None))
                    data.mfpt_to_B = torch.from_numpy(mfpt_log.astype(np.float32))

            # Store A/B masks for training
            data.A_mask = torch.from_numpy(A_sel)
            data.B_mask = torch.from_numpy(B_sel)

            # Metadata
            data.dps_dir = dps_key
            data.sequence = dps_dir.parent.name.replace("_nocap", "")
            data.n_nodes = Q.shape[0]

            data_list.append(data)
            print(f"  [ktn_dataset] {name}: N={Q.shape[0]}, "
                  f"E={edge_index.shape[1]}, |A|={A_sel.sum()}, |B|={B_sel.sum()}")

        print(f"[ktn_dataset] Built {len(data_list)} graphs.")
        self.save(data_list, self.processed_paths[0])


# ======================================================================
#  CLI for standalone processing
# ======================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build KTN PyG dataset.")
    parser.add_argument("--root", type=str, default="ktn_pyg_data",
                        help="Directory for cached PyG data.")
    parser.add_argument("--targets-csv", type=Path,
                        default=Path("GTcheck_micro_vs_coarse_T300K_full.csv"))
    parser.add_argument("--T", type=float, default=300.0)
    parser.add_argument("--no-node-targets", action="store_true",
                        help="Skip committor/MFPT computation (faster).")
    args = parser.parse_args()

    from config import BASE_DIR
    dataset = KTNDataset(
        root=args.root,
        base_dir=BASE_DIR,
        T=args.T,
        targets_csv=args.targets_csv,
        compute_node_targets=not args.no_node_targets,
    )
    print(f"Dataset: {len(dataset)} graphs")
    if len(dataset) > 0:
        print(f"  Node features: {dataset[0].x.shape}")
        print(f"  Edge features: {dataset[0].edge_attr.shape}")
        print(f"  Graph targets: {dataset[0].y}")
        if hasattr(dataset[0], "committor"):
            q = dataset[0].committor
            print(f"  Committor range: [{q.min():.4f}, {q.max():.4f}]")


if __name__ == "__main__":
    main()
