"""
markov_analysis.py

High-level analysis for a *single* DPS directory and temperature.

This script assumes that:
- build_markov_model.py has already been run for this DPS dir and temperature,
- so PyGT matrices live under markov_paths.pygt_dir.

It will:
- load B, tau, pi, Q
- if available, load A and B sets from min.A / min.B
- compute MFPTs A<->B and phenomenological rates with PyGT.stats
- compute barrier-graph and rate-based Dijkstra distances between A and B
- (optionally) compute full inter-microstate MFPT matrix with PyGT.mfpt
- dump a small JSON summary of metrics for later regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
from scipy.sparse import load_npz, issparse, diags

import PyGT  # type: ignore

from config import MarkovFilePaths, TEMPERATURES, N_EIGS, MAX_SOURCES_FOR_FULL_MFPT, RNG_SEED
from graph_distances import barrier_distances, rate_based_lengths
from stationary_point_io import read_min_ts


def _load_markov(markov_paths: MarkovFilePaths):
    """Load B, K, tau, Q, pi."""
    B = load_npz(markov_paths.B_path)
    K = load_npz(markov_paths.K_path)
    Q = load_npz(markov_paths.Q_path)
    tau = np.load(markov_paths.tau_path)
    pi = np.load(markov_paths.pi_path)
    return B, K, Q, tau, pi


def _ensure_generator(K, tau):
    """
    If for some reason Q_TxxxK.npz is missing, reconstruct generator
    Q = K - diag(1/tau). Uses column-sum convention (PyGT).
    """
    if issparse(K):
        Q = K - diags(1.0 / tau)
    else:
        Q = K - np.diag(1.0 / tau)
    return Q


def analyze_one(
    dps_dir: Path,
    T: float,
    compute_full_mfpt: bool = False,
) -> Dict[str, Any]:
    """
    Core analysis routine.

    Parameters
    ----------
    dps_dir : Path
        Directory containing min.data, ts.data, min.A, min.B, etc.
    T : float
        Temperature in Kelvin.
    compute_full_mfpt : bool
        If True, compute an N×N MFPT matrix (can be heavy!).

    Returns
    -------
    summary : dict
        Various scalar metrics for this (sequence, DPS run, T).
    """
    mp = MarkovFilePaths(dps_dir, T)
    mp.pygt_dir.mkdir(parents=True, exist_ok=True)

    B, K, Q, tau, pi = _load_markov(mp)

    # Safety: reconstruct Q if needed
    if Q is None or Q.shape != K.shape:
        Q = _ensure_generator(K, tau)

    N = pi.size
    total_escape_rates = 1.0 / tau
    avg_escape_rate = float(np.mean(total_escape_rates))

    # ----- A / B macrostates, if present -----
    # retained_mask has length = nnodes in min.data;
    # PyGT.io.load_ktn_AB will map that to A/B in retained indexing.
    retained = np.load(mp.retained_mask_path)
    A_vec, B_vec = PyGT.io.load_ktn_AB(str(dps_dir), retained=retained)
    has_AB = A_vec.sum() > 0 and B_vec.sum() > 0

    # ----- MFPTs and rates between A and B -----
    mfpt_AB = np.nan
    mfpt_BA = np.nan
    kF_AB = np.nan
    kF_BA = np.nan

    if has_AB:
        # Compute MFPT distribution moments using full generator Q
        moments, _ = PyGT.stats.compute_passage_stats(
            A_vec, B_vec, pi, Q, dopdf=False
        )
        # moments = [tau_BA, var_BA, tau_AB, var_AB]
        mfpt_BA = float(moments[0])
        mfpt_AB = float(moments[2])

        # Phenomenological rates and alternative MFPT definition
        rates = PyGT.stats.compute_rates(
            A_vec, B_vec, B, tau, pi, MFPTonly=False, fullGT=False, screen=False
        )
        # kF_* are the "forward" rates defined in the docs
        kF_AB = float(rates["kFAB"])
        kF_BA = float(rates["kFBA"])

    # ----- Spectral timescales (microscopic) -----
    from scipy.sparse.linalg import eigs as sparse_eigs

    # Q is column-sum-zero; eigenvalues of Q.T are the usual generator eigenvalues.
    evals, _ = sparse_eigs(Q.T, k=min(N_EIGS, N - 1), which="LR")
    lambdas = np.real(evals)
    # Sort by increasing |lambda| (lambda=0 is stationary)
    order = np.argsort(np.abs(lambdas))
    lambdas = lambdas[order]

    # Ignore lambda ~ 0, keep the slow non-zero modes
    nonzero = np.abs(lambdas) > 1e-12
    slow_lambdas = lambdas[nonzero]
    timescales = (1.0 / np.abs(slow_lambdas)).tolist()

    # ----- Barrier-based and rate-based distances between A and B -----
    avg_barrier_AB = np.nan
    min_barrier_AB = np.nan
    avg_rate_len_AB = np.nan
    min_rate_len_AB = np.nan

    if has_AB:
        # Use reactant-side weighting for averages
        rng = np.random.default_rng(RNG_SEED)

        A_indices = np.nonzero(A_vec)[0]
        B_indices = np.nonzero(B_vec)[0]

        # Barrier distances (undirected, sum of barrier heights)
        bdists = barrier_distances(dps_dir, mp, sources=A_indices)
        # bdists shape = (nA, N)
        # Take distances from A to nearest B
        AB_barrier = []
        for i, ai in enumerate(A_indices):
            # distances from this A to all B
            d_to_B = bdists[i, B_indices]
            AB_barrier.append(np.min(d_to_B))
        AB_barrier = np.asarray(AB_barrier)

        avg_barrier_AB = float(np.mean(AB_barrier))
        min_barrier_AB = float(np.min(AB_barrier))

        # Rate-based distances (directed, -log k_ij)
        kdists = rate_based_lengths(mp, sources=A_indices)
        AB_klen = []
        for i, ai in enumerate(A_indices):
            d_to_B = kdists[i, B_indices]
            AB_klen.append(np.min(d_to_B))
        AB_klen = np.asarray(AB_klen)
        avg_rate_len_AB = float(np.mean(AB_klen))
        min_rate_len_AB = float(np.min(AB_klen))

    # ----- Optional full MFPT matrix -----
    full_mfpt_shape = None
    if compute_full_mfpt and MAX_SOURCES_FOR_FULL_MFPT is None:
        # Really compute all pairs
        tauM = PyGT.mfpt.full_MFPT_matrix(B, tau, pool_size=1, screen=False)
        np.save(mp.mfpt_matrix_path, tauM)
        full_mfpt_shape = list(tauM.shape)
        del tauM

    summary: Dict[str, Any] = {
        "sequence_dir": dps_dir.parent.name,
        "dps_dir": dps_dir.name,
        "temperature_K": float(T),
        "N_states": int(N),
        "avg_escape_rate": avg_escape_rate,
        "has_AB": bool(has_AB),
        "N_A": int(A_vec.sum()) if has_AB else 0,
        "N_B": int(B_vec.sum()) if has_AB else 0,
        "mfpt_AB": mfpt_AB,
        "mfpt_BA": mfpt_BA,
        "kF_AB": kF_AB,
        "kF_BA": kF_BA,
        "slow_timescales": timescales,  # list of floats
        "avg_barrier_AB": avg_barrier_AB,
        "min_barrier_AB": min_barrier_AB,
        "avg_rate_length_AB": avg_rate_len_AB,
        "min_rate_length_AB": min_rate_len_AB,
        "full_mfpt_shape": full_mfpt_shape,
    }

    # Cache to JSON
    with open(mp.summary_json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Markov / MFPT / distance analysis for one DPS directory."
    )
    parser.add_argument(
        "dps_dir",
        type=str,
        help="Path to DPS directory containing min.data, ts.data, etc.",
    )
    parser.add_argument(
        "--T",
        type=float,
        default=TEMPERATURES[0],
        help="Temperature in Kelvin (must match build_markov_model.py).",
    )
    parser.add_argument(
        "--full-mfpt",
        action="store_true",
        help="Also compute full N×N MFPT matrix (can be expensive).",
    )
    args = parser.parse_args()

    dps_dir = Path(args.dps_dir).resolve()
    summary = analyze_one(dps_dir, args.T, compute_full_mfpt=args.full_mfpt)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
