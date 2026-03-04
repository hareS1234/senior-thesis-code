#!/usr/bin/env python
"""
io_markov.py

Helpers for loading microscopic or coarse-grained Markov models
built by build_markov_model.py and coarse_grain_markov.py, and for
getting A/B macrostates in the appropriate indexing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp

from PyGT.io import load_ktn_AB  # uses min.A/min.B + retained mask


def temp_tag(T: float) -> str:
    """Format temperature tag, e.g. 300.0 -> 'T300K'."""
    return f"T{int(round(T))}K"


def markov_dir_for_T(data_dir: Path, T: float) -> Path:
    """Directory where build_markov_model.py writes its outputs."""
    return data_dir / f"markov_{temp_tag(T)}"


def coarse_dir_for_T(markov_dir: Path, T: float) -> Path:
    """Directory where coarse_grain_markov.py writes GT outputs."""
    tag = temp_tag(T)
    return markov_dir / f"GT_kept_{tag}"


def _load_sparse(path: Path) -> sp.csr_matrix:
    mat = sp.load_npz(path)
    if not isinstance(mat, sp.csr_matrix):
        mat = mat.tocsr()
    return mat


def load_markov(
    data_dir: Path,
    T: float,
    coarse: bool = False,
) -> Tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix, np.ndarray, np.ndarray]:
    """
    Load B, K, Q, tau, pi for microscopic or coarse-grained model.

    Parameters
    ----------
    data_dir : Path
        DPS directory containing min.data, ts.data, and markov_TxxxK/.
    T : float
        Temperature in Kelvin (must match build_markov_model.py).
    coarse : bool
        If False: microscopic model in markov_TxxxK/.
        If True : coarse model in markov_TxxxK/GT_kept_TxxxK/.

    Returns
    -------
    B : csr_matrix
        Branching probability matrix (columns sum to 1, zero diagonals).
    K : csr_matrix
        Off-diagonal rate matrix k_{i<-j}.
    Q : csr_matrix
        Generator matrix (columns sum to zero).
    tau : ndarray (N,)
        Mean waiting times in each state.
    pi : ndarray (N,)
        Stationary distribution.
    """
    tag = temp_tag(T)
    markov_dir = markov_dir_for_T(data_dir, T)

    if coarse:
        eff_dir = coarse_dir_for_T(markov_dir, T)
        B_path = eff_dir / f"B_eff_{tag}.npz"
        K_path = eff_dir / f"K_eff_{tag}.npz"
        Q_path = eff_dir / f"Q_eff_{tag}.npz"
        tau_path = eff_dir / f"tau_eff_{tag}.npy"
        pi_path = eff_dir / f"pi_eff_{tag}.npy"
    else:
        B_path = markov_dir / f"B_{tag}.npz"
        K_path = markov_dir / f"K_{tag}.npz"
        Q_path = markov_dir / f"Q_{tag}.npz"
        tau_path = markov_dir / f"tau_{tag}.npy"
        pi_path = markov_dir / f"pi_{tag}.npy"

    B = _load_sparse(B_path)
    tau = np.load(tau_path)
    pi = np.load(pi_path)

    if K_path.exists():
        K = _load_sparse(K_path)
    elif Q_path.exists():
        # Reconstruct K from Q: K_ij = Q_ij for i!=j, K_ii = 0
        Q_tmp = _load_sparse(Q_path)
        K = Q_tmp.copy()
        K.setdiag(0)
        K.eliminate_zeros()
    else:
        raise FileNotFoundError(f"Neither K nor Q found in {K_path.parent}")

    if Q_path.exists():
        Q = _load_sparse(Q_path)
    else:
        # Reconstruct generator: Q = K - diag(1/tau)
        escape = 1.0 / tau
        Q = (K - sp.diags(escape, offsets=0, format="csr")).tocsr()

    return B, K, Q, tau, pi


def load_AB_selectors(
    data_dir: Path,
    T: float,
    coarse: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get A/B macrostates for the microscopic or coarse model.

    Minima in A and B are specified by PATHSAMPLE files min.A and min.B.
    For the microscopic model, we use PyGT.io.load_ktn_AB with the same
    retained mask as build_markov_model.py.

    For the coarse model, we map those A/B labels onto the retained
    minima in the GT-reduced network using the original minima IDs.

    Returns
    -------
    A_sel, B_sel : ndarray[bool] or (None, None)
        Boolean selectors in the *current* indexing (microscopic or coarse).
        If min.A/min.B are missing, returns (None, None).
    """
    tag = temp_tag(T)
    markov_dir = markov_dir_for_T(data_dir, T)
    retained_path = markov_dir / f"retained_mask_{tag}.npy"

    if not retained_path.exists():
        # No Markov model yet
        return None, None

    retained = np.load(retained_path)

    # Check if min.A and min.B exist
    if not (data_dir / "min.A").exists() or not (data_dir / "min.B").exists():
        return None, None

    # Microscopic A/B selectors (length N_micro)
    A_micro, B_micro = load_ktn_AB(str(data_dir), retained=retained)

    if not coarse:
        return A_micro, B_micro

    # Map to coarse indices
    eff_dir = coarse_dir_for_T(markov_dir, T)
    orig_micro_ids_path = markov_dir / f"original_min_ids_{tag}.npy"
    orig_eff_ids_path = eff_dir / f"orig_min_ids_eff_{tag}.npy"

    if not (orig_micro_ids_path.exists() and orig_eff_ids_path.exists()):
        return None, None

    orig_micro_ids = np.load(orig_micro_ids_path)  # length N_micro
    orig_eff_ids = np.load(orig_eff_ids_path)      # length N_eff

    id_to_micro = {int(mid): int(i) for i, mid in enumerate(orig_micro_ids)}

    N_eff = len(orig_eff_ids)
    A_eff = np.zeros(N_eff, dtype=bool)
    B_eff = np.zeros(N_eff, dtype=bool)

    for i_eff, orig_id in enumerate(orig_eff_ids):
        idx = id_to_micro.get(int(orig_id), None)
        if idx is None:
            continue
        A_eff[i_eff] = A_micro[idx]
        B_eff[i_eff] = B_micro[idx]

    return A_eff, B_eff
