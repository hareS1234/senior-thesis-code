#!/usr/bin/env python
"""
build_markov_model.py

Use PyGT to build a microscopic continuous-time Markov model for a
PATHSAMPLE DPS directory at a given temperature.

Outputs go to:
    <DPS_DIR>/markov_TxxxK/

and include B, K, Q, tau, pi, energies, entropies, etc.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from PyGT.io import load_ktn, load_ktn_AB
from PyGT.tools import check_detailed_balance

from config import MarkovFilePaths

KB_KCAL_MOLK = 0.0019872041  # kcal/mol/K


def build_markov_model(
    data_dir: Path,
    temperature: float,
    energy_unit: str = "kcal/mol",
    kB: float = KB_KCAL_MOLK,
    Nmax: int | None = None,
    Emax: float | None = None,
    screen: bool = True,
) -> None:
    data_dir = data_dir.resolve()
    mp = MarkovFilePaths(data_dir, temperature)
    markov_dir = mp.markov_dir
    markov_dir.mkdir(parents=True, exist_ok=True)

    beta = 1.0 / (kB * temperature)
    tag = mp.temp_tag

    if screen:
        print(f"=== build_markov_model: {data_dir}, T={temperature:.1f} K ===")
        print(f"Output dir: {markov_dir}")
        print(f"beta = {beta:.6f} 1/{energy_unit}")

    # 1. Use PyGT to get B, K, tau, energies, entropies
    B, K, tau, N, energies, entropies, Emin, retained = load_ktn(
        str(data_dir), beta=beta, Nmax=Nmax, Emax=Emax, screen=screen
    )
    B = B.tocsr()
    K = K.tocsr()

    if screen:
        print(f"N = {N}, Emin shift = {Emin:.4f} {energy_unit}")
        print(f"Retained minima: {retained.sum()} of {retained.size}")

    # 2. Build generator Q (columns sum to zero)
    escape_rates = 1.0 / tau
    Lambda = sp.diags(escape_rates, offsets=0, format="csr")
    Q = (K - Lambda).tocsr()

    col_sums = np.array(Q.sum(axis=0)).ravel()
    if screen:
        print(f"max |column sum of Q| = {np.max(np.abs(col_sums)):.3e}")

    # 3. Stationary distribution pi ∝ exp(-β u_i + s_i)
    log_pi = -beta * energies + entropies
    log_pi -= np.max(log_pi)
    pi = np.exp(log_pi)
    pi /= pi.sum()

    if screen:
        print(f"pi min/max = {pi.min():.3e} / {pi.max():.3e}")

    ok = check_detailed_balance(pi, K)
    if screen:
        print(f"Detailed balance (K, pi): {ok}")

    res = Q @ pi
    if screen:
        print(f"||Q*pi||_inf = {np.max(np.abs(res)):.3e}")

    # 4. Optional A/B sets (if min.A/min.B exist)
    minA = data_dir / "min.A"
    minB = data_dir / "min.B"
    A_states = None
    B_states = None
    if minA.exists() and minB.exists():
        A_states, B_states = load_ktn_AB(str(data_dir), retained=retained)

    # 5. Save everything
    def save_sparse(mat: sp.spmatrix, name: str):
        sp.save_npz(markov_dir / f"{name}_{tag}.npz", mat)

    save_sparse(B, "B")
    save_sparse(K, "K")
    save_sparse(Q, "Q")

    np.save(markov_dir / f"tau_{tag}.npy", tau)
    np.save(markov_dir / f"pi_{tag}.npy", pi)
    np.save(markov_dir / f"energies_{tag}.npy", energies)
    np.save(markov_dir / f"entropies_{tag}.npy", entropies)
    np.save(markov_dir / f"retained_mask_{tag}.npy", retained)

    orig_ids = np.nonzero(retained)[0] + 1  # 1-based PATHSAMPLE minima IDs
    np.save(markov_dir / f"original_min_ids_{tag}.npy", orig_ids)

    if A_states is not None:
        np.save(markov_dir / f"A_states_{tag}.npy", A_states)
        np.save(markov_dir / f"B_states_{tag}.npy", B_states)

    meta = {
        "temperature_K": float(temperature),
        "beta": float(beta),
        "energy_unit": energy_unit,
        "kB": float(kB),
        "N": int(N),
    }
    np.save(markov_dir / f"meta_{tag}.npy", meta)

    if screen:
        print(f"[build_markov_model] Saved Markov model in {markov_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Build microscopic CTMC from PATHSAMPLE DPS data via PyGT."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="DPS directory with min.data, ts.data, etc.",
    )
    parser.add_argument(
        "--T",
        "--temperature",
        type=float,
        required=True,
        help="Temperature in Kelvin.",
    )
    parser.add_argument(
        "--Nmax",
        type=int,
        default=None,
        help="Max number of minima (PyGT load_ktn).",
    )
    parser.add_argument(
        "--Emax",
        type=float,
        default=None,
        help="Energy cutoff (PyGT load_ktn).",
    )
    parser.add_argument(
        "--kB",
        type=float,
        default=KB_KCAL_MOLK,
        help="Boltzmann constant in energy units/K.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output.",
    )
    args = parser.parse_args()

    build_markov_model(
        data_dir=args.data_dir,
        temperature=args.T,
        kB=args.kB,
        Nmax=args.Nmax,
        Emax=args.Emax,
        screen=not args.quiet,
    )


if __name__ == "__main__":
    main()
