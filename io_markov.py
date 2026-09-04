#!/usr/bin/env python
"""Load the microscopic/coarse Markov files and line up their A/B states."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import scipy.sparse as sp

from PyGT.io import load_ktn_AB


def temp_tag(T: float) -> str:
    """Turn 300.0 into the ``T300K`` filename tag."""
    return f"T{int(round(T))}K"


def markov_dir_for_T(data_dir: Path, T: float) -> Path:
    """Microscopic model folder for this temperature."""
    return data_dir / f"markov_{temp_tag(T)}"


def coarse_dir_for_T(markov_dir: Path, T: float) -> Path:
    """GT-kept folder for this temperature."""
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
    """Load ``B, K, Q, tau, pi`` from the micro or GT-kept folder."""
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

        Q_tmp = _load_sparse(Q_path)
        K = Q_tmp.copy()
        K.setdiag(0)
        K.eliminate_zeros()
    else:
        raise FileNotFoundError(f"Neither K nor Q found in {K_path.parent}")

    if Q_path.exists():
        Q = _load_sparse(Q_path)
    else:

        escape = 1.0 / tau
        Q = (K - sp.diags(escape, offsets=0, format="csr")).tocsr()

    return B, K, Q, tau, pi


def load_AB_selectors(
    data_dir: Path,
    T: float,
    coarse: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Map min.A/min.B into whichever model indexing is being loaded."""
    tag = temp_tag(T)
    markov_dir = markov_dir_for_T(data_dir, T)
    retained_path = markov_dir / f"retained_mask_{tag}.npy"

    if not retained_path.exists():

        return None, None

    retained = np.load(retained_path)


    if not (data_dir / "min.A").exists() or not (data_dir / "min.B").exists():
        return None, None


    A_micro, B_micro = load_ktn_AB(str(data_dir), retained=retained)

    if not coarse:
        return A_micro, B_micro


    eff_dir = coarse_dir_for_T(markov_dir, T)


    A_direct = eff_dir / f"A_states_{tag}.npy"
    B_direct = eff_dir / f"B_states_{tag}.npy"
    if A_direct.exists() and B_direct.exists():
        return np.load(A_direct), np.load(B_direct)


    orig_micro_ids_path = markov_dir / f"original_min_ids_{tag}.npy"

    orig_eff_ids_path = eff_dir / f"original_min_ids_eff_{tag}.npy"
    if not orig_eff_ids_path.exists():
        orig_eff_ids_path = eff_dir / f"orig_min_ids_eff_{tag}.npy"

    if not (orig_micro_ids_path.exists() and orig_eff_ids_path.exists()):
        return None, None

    orig_micro_ids = np.load(orig_micro_ids_path)
    orig_eff_ids = np.load(orig_eff_ids_path)

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
