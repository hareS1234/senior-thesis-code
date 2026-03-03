#!/usr/bin/env python
"""
coarse_grain_markov.py

Use PyGT.GT.blockGT to perform graph transformation on a microscopic
Markov model (B, tau), eliminating all minima EXCEPT those listed in
keep_minima.txt.

Inputs:
    markov_dir/      (from build_markov_model.py)
        B_TxxxK.npz
        K_TxxxK.npz
        tau_TxxxK.npy
        pi_TxxxK.npy
        original_min_ids_TxxxK.npy
        retained_mask_TxxxK.npy

    keep_minima.txt  (in DPS directory, original 1-based min IDs to keep)

Outputs:
    markov_dir/GT_kept_TxxxK/
        B_eff_TxxxK.npz
        K_eff_TxxxK.npz
        Q_eff_TxxxK.npz
        tau_eff_TxxxK.npy
        pi_eff_TxxxK.npy
        orig_min_ids_eff_TxxxK.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from PyGT.GT import blockGT
from PyGT.tools import check_detailed_balance

from config import MarkovFilePaths


def _read_keep_list(path: Path) -> np.ndarray:
    """Read keep_minima.txt -> array of 1-based minima IDs to keep."""
    vals = np.loadtxt(path, dtype=int)
    if vals.ndim == 0:
        vals = np.array([int(vals)], dtype=int)
    return vals


def coarse_grain(
    markov_dir: Path,
    T: float,
    keep_file: Path,
    output_dir: Path | None = None,
    block_size: int = 20,
    screen: bool = True,
) -> None:
    mp = MarkovFilePaths(markov_dir.parent, T)
    tag = mp.temp_tag
    markov_dir = mp.markov_dir  # ensure consistency

    if output_dir is None:
        output_dir = markov_dir / f"GT_kept_{tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if screen:
        print(f"=== coarse_grain_markov: {markov_dir}, T={T:.1f} K ===")
        print(f"Keeping minima from {keep_file}")
        print(f"Output dir: {output_dir}")

    # Microscopic model
    B = sp.load_npz(mp.B_path).tocsr()
    K = sp.load_npz(mp.K_path).tocsr()
    tau = np.load(mp.tau_path)
    pi = np.load(mp.pi_path)
    orig_ids = np.load(mp.orig_ids_path)  # length N_micro

    N = B.shape[0]
    if screen:
        print(f"Microscopic N = {N}")

    # Which minima to keep? map original IDs -> microscopic indices
    keep_orig = _read_keep_list(keep_file)
    id_to_idx = {int(mid): int(i) for i, mid in enumerate(orig_ids)}

    keep_mask = np.zeros(N, dtype=bool)
    for mid in keep_orig:
        idx = id_to_idx.get(int(mid))
        if idx is not None:
            keep_mask[idx] = True

    rm_vec = ~keep_mask
    if not keep_mask.any():
        raise RuntimeError("Keep list does not intersect microscopic minima!")

    if screen:
        print(f"Keeping {keep_mask.sum()} minima; removing {rm_vec.sum()}")

    # Run GT
    B_eff, tau_eff, K_eff = blockGT(
        rm_vec, B, tau, block=block_size, rates=True, screen=screen
    )

    # PyGT often returns numpy.matrix; convert to proper sparse CSR
    import numpy as _np
    if isinstance(B_eff, _np.matrix):
        B_eff = _np.array(B_eff)
    if isinstance(K_eff, _np.matrix):
        K_eff = _np.array(K_eff)

    if not sp.isspmatrix(B_eff):
        B_eff = sp.csr_matrix(B_eff)
    if not sp.isspmatrix(K_eff):
        K_eff = sp.csr_matrix(K_eff)

    N_eff = B_eff.shape[0]
    if screen:
        print(f"Reduced N_eff = {N_eff}")

    # Construct Q_eff and pi_eff
    escape_eff = 1.0 / tau_eff
    Lambda_eff = sp.diags(escape_eff, offsets=0, format="csr")
    Q_eff = (K_eff - Lambda_eff).tocsr()

    pi_eff = pi[keep_mask]
    pi_eff /= pi_eff.sum()

    ok = check_detailed_balance(pi_eff, K_eff)
    if screen:
        print(f"Detailed balance (coarse): {ok}")

    # Mapping reduced indices -> original minima IDs
    kept_orig_ids = orig_ids[keep_mask]

    # Save
    def save_sparse(mat: sp.spmatrix, name: str):
        sp.save_npz(output_dir / f"{name}_{tag}.npz", mat)

    save_sparse(B_eff, "B_eff")
    save_sparse(K_eff, "K_eff")
    save_sparse(Q_eff, "Q_eff")

    np.save(output_dir / f"tau_eff_{tag}.npy", tau_eff)
    np.save(output_dir / f"pi_eff_{tag}.npy", pi_eff)
    np.save(output_dir / f"orig_min_ids_eff_{tag}.npy", kept_orig_ids)

    if screen:
        print(f"[coarse_grain_markov] Saved coarse model in {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Coarse-grain Markov model via PyGT graph transformation."
    )
    parser.add_argument(
        "--markov-dir",
        type=Path,
        required=True,
        help="Directory with microscopic Markov model (markov_TxxxK). "
             "NOTE: this is just used to locate the DPS dir; consistency "
             "with config.MarkovFilePaths is enforced.",
    )
    parser.add_argument(
        "--T",
        type=float,
        required=True,
        help="Temperature in Kelvin.",
    )
    parser.add_argument(
        "--keep-file",
        type=Path,
        required=True,
        help="keep_minima.txt with original 1-based PATHSAMPLE IDs to keep.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for output directory (default: markov_TxxxK/GT_kept_TxxxK).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=20,
        help="Block size for blockGT (performance knob).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output.",
    )
    args = parser.parse_args()

    coarse_grain(
        markov_dir=args.markov_dir,
        T=args.T,
        keep_file=args.keep_file,
        output_dir=args.output_dir,
        block_size=args.block_size,
        screen=not args.quiet,
    )


if __name__ == "__main__":
    main()
