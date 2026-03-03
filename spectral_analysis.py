#!/usr/bin/env python
"""
spectral_analysis.py

Given a generator Q(T) stored as a sparse matrix, compute the slowest
relaxation timescales via its eigenvalues.
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def analyse_spectrum(Q_file: Path, k: int, output_prefix: Path | None = None):
    Q = sp.load_npz(Q_file).tocsr()
    N = Q.shape[0]
    if output_prefix is None:
        output_prefix = Q_file.with_suffix("")

    print(f"Loaded Q from {Q_file}, N = {N}")

    # Our Q has columns summing to zero (Q_ij = k_{i<-j}, diag negative).
    # For right eigenvectors we want row-sum-zero form, so we diagonalise Q^T.
    Q_row = Q.T.tocsr()

    # Compute k eigenvalues with largest real part (closest to 0).
    # lambda_0 ≈ 0, others negative.
    print(f"Computing {k} eigenvalues of Q^T closest to 0...")
    eigvals, eigvecs = spla.eigs(Q_row, k=k, which="LR")  # largest real part

    # Sort by real part descending (0, then slowest modes)
    order = np.argsort(-eigvals.real)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Timescales tau_n = -1 / Re(lambda_n) for n>=1 (ignore lambda_0=0)
    timescales = np.empty(k)
    timescales[0] = np.inf
    for n in range(1, k):
        lam = eigvals[n]
        timescales[n] = -1.0 / lam.real

    np.save(f"{output_prefix}_eigvals.npy", eigvals)
    np.save(f"{output_prefix}_eigvecs.npy", eigvecs)
    np.save(f"{output_prefix}_timescales.npy", timescales)

    print("Eigenvalues (real parts):")
    for n in range(k):
        print(f"  n={n:2d}: Re(lambda)={eigvals[n].real: .4e}, Im={eigvals[n].imag: .4e}, tau={timescales[n]:.4e}")

    print(f"Saved eigvals/eigvecs/timescales with prefix {output_prefix}")


def main():
    parser = argparse.ArgumentParser(
        description="Spectral analysis of sparse generator Q(T)"
    )
    parser.add_argument(
        "--Q-file",
        type=Path,
        required=True,
        help="Path to Q_TxxxK.npz (or Q_eff_TxxxK.npz)",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="Number of eigenvalues/vectors to compute (default 10)",
    )
    args = parser.parse_args()
    analyse_spectrum(args.Q_file, args.k)


if __name__ == "__main__":
    main()
