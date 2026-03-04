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
    Compute the *forward committor* q_i^+ for each node:

        q_i^+ = P(reach B before A | start at i).

    With our convention, the generator `Q` acts on *probability vectors* via
    dp/dt = Q p, so columns sum to zero.  The associated *backward*
    (row-sum-zero) generator is G = Q^T.

    Boundary conditions:
        q_A = 0,  q_B = 1

    Interior equation (backward equation):
        G_II q_I = - G_IB 1

    Returns
    -------
    q : (N,) ndarray in [0, 1], or None on failure.
    """
    N = Q.shape[0]
    A_sel = np.asarray(A_sel, dtype=bool)
    B_sel = np.asarray(B_sel, dtype=bool)

    if A_sel.shape[0] != N or B_sel.shape[0] != N:
        raise ValueError("A_sel/B_sel must be boolean arrays of length N=Q.shape[0].")

    B_idx = np.where(B_sel)[0]
    I_mask = ~(A_sel | B_sel)
    I_idx = np.where(I_mask)[0]

    # Trivial case: no interior states.
    if I_idx.size == 0:
        q = np.zeros(N, dtype=float)
        q[B_sel] = 1.0
        return q

    # Backward generator (row-sum-zero)
    G = Q.T.tocsr()

    # Sub-blocks on interior nodes
    G_II = G[np.ix_(I_idx, I_idx)].tocsc()
    G_IB = G[np.ix_(I_idx, B_idx)]

    rhs = -G_IB @ np.ones(B_idx.size, dtype=float)

    try:
        q_I = spsolve(G_II, rhs)
    except Exception:
        return None

    # Numerical cleanup
    q_I = np.clip(q_I, 0.0, 1.0)

    q = np.zeros(N, dtype=float)
    q[B_sel] = 1.0
    q[I_idx] = q_I
    return q


def compute_mfpt_to_B(
    Q: csr_matrix,
    B_sel: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute the mean first-passage time (MFPT) from each node to set B.

    With the column-sum-zero convention dp/dt = Q p, the backward generator is
    G = Q^T. The MFPT m solves, for i in I = complement(B):

        (G m)_i = -1,   with boundary m_B = 0.

    Returns
    -------
    m : (N,) ndarray (non-negative), or None on failure.
    """
    N = Q.shape[0]
    B_sel = np.asarray(B_sel, dtype=bool)
    if B_sel.shape[0] != N:
        raise ValueError("B_sel must be a boolean array of length N=Q.shape[0].")

    I_mask = ~B_sel
    I_idx = np.where(I_mask)[0]

    if I_idx.size == 0:
        return np.zeros(N, dtype=float)

    G = Q.T.tocsr()
    G_II = G[np.ix_(I_idx, I_idx)].tocsc()

    rhs = -np.ones(I_idx.size, dtype=float)

    try:
        m_I = spsolve(G_II, rhs)
    except Exception:
        return None

    m = np.zeros(N, dtype=float)
    # Clip tiny negative values from numerical error
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
    K: Optional[csr_matrix] = None,
    energies: Optional[np.ndarray] = None,
    entropies: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Build node feature matrix [N, D_node].

    Features:
        0: log(pi)          (standardized within graph)
        1: log(tau)          (standardized within graph)
        2: energy            (standardized within graph, or 0)
        3: entropy           (standardized within graph, or 0)
        4: is_A              (binary)
        5: is_B              (binary)
        6: mean_log_rate     (global scalar, same for all nodes)
        7: std_log_rate      (global scalar, same for all nodes)
        8: mean_log_tau      (global scalar, same for all nodes)

    Columns 6-8 are *not* standardized within the graph.  They encode the
    absolute kinetic scale of this network, which is critical for
    graph-level MFPT prediction across different sequences.
    """
    N = pi.size
    feats = np.zeros((N, 9), dtype=np.float32)

    log_pi = np.log(np.clip(pi, 1e-300, None))
    log_tau = np.log(np.clip(tau, 1e-300, None))

    feats[:, 0] = log_pi
    feats[:, 1] = log_tau

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

    # Global-scale features (constant across all nodes in this graph,
    # NOT standardized, so the model can learn cross-graph scale differences)
    if K is not None:
        K_coo = K.tocoo()
        off_diag = K_coo.data[K_coo.row != K_coo.col]
        if off_diag.size > 0:
            log_rates = np.log(np.clip(off_diag, 1e-300, None))
            feats[:, 6] = float(np.mean(log_rates))
            feats[:, 7] = float(np.std(log_rates))
        # else columns stay 0

    finite_log_tau = log_tau[np.isfinite(log_tau)]
    if finite_log_tau.size > 0:
        feats[:, 8] = float(np.mean(finite_log_tau))

    return torch.from_numpy(feats)


def build_edge_features(
    K: csr_matrix,
    B_mat: csr_matrix,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build (edge_index, edge_attr) for PyTorch Geometric.

    Conventions
    ----------
    K[i, j] = k_{i <- j} is the *rate into i from j* (so columns are sources).
    Therefore each nonzero off-diagonal entry K[i, j] corresponds to a directed
    edge (source=j) -> (target=i).

    Edge features (D_edge = 4)
    -------------------------
        0: log(k_{i<-j})   forward log-rate
        1: log(k_{j<-i})   reverse log-rate (0 if no reverse edge)
        2: B_{i<-j}        branching probability for this jump
        3: has_reverse      1.0 if reverse edge j<-i exists, else 0.0

    Notes
    -----
    We standardize the two log-rate columns *within each graph* to stabilize
    training across networks with different absolute prefactors.  The binary
    ``has_reverse`` feature lets the model distinguish "no reverse edge" from
    "very small reverse rate" after standardization shifts the sentinel 0.
    """
    K_coo = K.tocoo()
    mask = K_coo.row != K_coo.col  # off-diagonal only

    rows = K_coo.row[mask].astype(np.int64)   # target i
    cols = K_coo.col[mask].astype(np.int64)   # source j
    rates = K_coo.data[mask].astype(float)

    # PyG edge_index uses [2, E] with (source, target)
    edge_index = torch.tensor(np.vstack([cols, rows]), dtype=torch.long)

    n_edges = rows.size
    edge_attr = np.zeros((n_edges, 4), dtype=np.float32)

    # Forward log-rate: log(K[i,j])
    edge_attr[:, 0] = np.log(np.clip(rates, 1e-300, None))

    # Reverse log-rate: log(K[j,i]) where present; else 0 by convention.
    K_csr = K.tocsr()
    rev_rates = np.asarray(K_csr[cols, rows]).ravel().astype(float)
    rev_log = np.zeros(n_edges, dtype=np.float32)
    has_reverse = rev_rates > 0
    rev_log[has_reverse] = np.log(np.clip(rev_rates[has_reverse], 1e-300, None))
    edge_attr[:, 1] = rev_log

    # Branching probability for the same directed edge j->i is B[i,j]
    B_csr = B_mat.tocsr()
    b_vals = np.asarray(B_csr[rows, cols]).ravel().astype(float)
    edge_attr[:, 2] = b_vals.astype(np.float32)

    # Binary indicator for reverse edge existence
    edge_attr[:, 3] = has_reverse.astype(np.float32)

    # Standardize forward log-rates (column 0) over all edges.
    fwd = edge_attr[:, 0]
    finite_fwd = fwd[np.isfinite(fwd)]
    if finite_fwd.size > 1 and finite_fwd.std() > 0:
        edge_attr[:, 0] = (fwd - finite_fwd.mean()) / finite_fwd.std()
    else:
        edge_attr[:, 0] = 0.0

    # Standardize reverse log-rates (column 1) only over edges that
    # actually have a reverse edge, so the sentinel 0 for missing
    # reverse edges doesn't pollute the mean/std.  Edges without a
    # reverse edge stay at 0; the has_reverse flag (column 3) lets the
    # model distinguish them.
    rev_mask = has_reverse
    if rev_mask.sum() > 1:
        rev_vals = edge_attr[rev_mask, 1]
        mu, sigma = rev_vals.mean(), rev_vals.std()
        if sigma > 0:
            edge_attr[rev_mask, 1] = (rev_vals - mu) / sigma
        else:
            edge_attr[rev_mask, 1] = 0.0
    # Missing-reverse edges keep their 0.0 sentinel

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

    # Bump this when node/edge feature definitions change to invalidate cache.
    _FEATURE_VERSION = "v3"  # v3: fixed reverse-edge standardization + global-scale features

    @property
    def processed_file_names(self):
        # Cache key includes build settings to avoid stale dataset reuse.
        tag = temp_tag(self.T)
        node_tag = "with_node_targets" if self.compute_node_targets else "no_node_targets"
        tgt_tag = self.targets_csv.stem if self.targets_csv is not None else "no_targets"
        base_tag = self.base_dir.name if self.base_dir is not None else "default_base"

        def _safe(s: str) -> str:
            return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)

        fname = (f"ktn_data_{_safe(base_tag)}_{tag}_{node_tag}"
                 f"_{_safe(tgt_tag)}_{self._FEATURE_VERSION}.pt")
        return [fname]

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

        dps_dirs = iter_dps_dirs(self.base_dir)
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
            x = build_node_features(pi, tau, A_sel, B_sel, K, energies, entropies)
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
                    mfpt_log = np.log10(np.clip(mfpt, 1e-12, None))  # avoid -inf at B nodes
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
