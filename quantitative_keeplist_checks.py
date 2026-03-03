#!/usr/bin/env python
"""
quantitative_keeplist_checks.py

Quantitative robustness checks for basin-based keep lists:

For a given DPS directory:
  - loop over several ΔE_cut values
  - for each, build a basin-based keep list
  - run your existing KTN + graph transformation pipeline to obtain Q_eff
  - compute MFPT_A->B, MFPT_B->A, and a few slow relaxation times
  - write a CSV summarizing how these quantities depend on ΔE_cut.

This script assumes you already have working code that:
  (1) builds a microscopic KTN from PATHSAMPLE outputs, and
  (2) applies graph transformation given a set of minima to keep.

You will need to hook those pieces into the `build_Qeff_for_deltaE(...)`
function below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Dict, Any, Tuple

import numpy as np

from generate_basin_keep_lists import build_basin_keep_set
from ktn_utils import compute_mfpt_from_Q, leading_relaxation_times


# --------------------------------------------------------------------
# 1. Utility: read min.A / min.B as sets of minima indices (0-based)
# --------------------------------------------------------------------
def read_min_list(path: Path) -> np.ndarray:
    """
    Read PATHSAMPLE-style min.A or min.B list.

    Returns a numpy array of 0-based indices.
    """
    if not path.exists():
        return np.array([], dtype=int)

    ids = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                mid = int(parts[0])
                ids.append(mid - 1)  # convert to 0-based
            except ValueError:
                continue
    return np.asarray(ids, dtype=int)


# --------------------------------------------------------------------
# 2. Hook: build Q_eff for a given ΔE_cut
# --------------------------------------------------------------------
def build_Qeff_for_deltaE(
    dps_dir: Path,
    deltaE_cut: float,
    E_window: float,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a coarse-grained generator Q_eff for a given ΔE_cut.

    You MUST adapt this to your own KTN + GT pipeline. The skeleton is:

      1) Build keep_ids via basin-based scheme.
      2) Load microscopic KTN from PATHSAMPLE (using your PyGT wrapper).
      3) Apply graph transformation with keep_ids as the kept set.
      4) Return Q_eff (row-generator), and the mapping from kept indices
         back to original minima indices (if needed).

    For now, we return only Q_eff; we assume states are in the same order
    as keep_ids.

    Replace the `raise NotImplementedError` block with your actual calls.
    """

    # Step 1: basin-based keep list (1-based -> 0-based conversion inside)
    keep_ids_1based = build_basin_keep_set(
        data_dir=dps_dir,
        deltaE_cut=deltaE_cut,
        E_window=E_window,
    )
    # convert to 0-based
    keep_ids = np.array(keep_ids_1based, dtype=int) - 1

    # --- STEP 2+3: INSERT YOUR EXISTING CODE HERE -------------------
    # Pseudocode (you will replace with real calls):

    # from your_ktn_module import load_ktn_from_pathsample, apply_blockGT

    # B, tau, pi = load_ktn_from_pathsample(dps_dir, temperature)
    # Q_micro = build_row_generator(B, tau)  # or use PyGT directly
    # Q_eff, kept_order = apply_blockGT(Q_micro, keep_ids)
    #
    # Make sure Q_eff is a ROW generator: rows sum to zero, p^T Q.
    #
    # return Q_eff, kept_order, keep_ids

    raise NotImplementedError(
        "You need to hook in your existing KTN + GT code in build_Qeff_for_deltaE."
    )

    # dummy to satisfy type checker; remove after implementing
    # return Q_eff, kept_order, keep_ids


# --------------------------------------------------------------------
# 3. Main comparison loop
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Quantitative robustness checks for basin-based keep lists."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="DPS directory containing min.data, ts.data, min.A/B, etc.",
    )
    parser.add_argument(
        "--deltaE-grid",
        type=str,
        default="10,15,20,25,30,40",
        help="Comma-separated list of ΔE_cut values to test.",
    )
    parser.add_argument(
        "--E-window",
        type=float,
        default=3.0,
        help="Energy window (E - Emin <= E_window) used in basin-based keep list.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=300.0,
        help="Temperature in K (pass through to your KTN builder).",
    )
    parser.add_argument(
        "--n-relax",
        type=int,
        default=5,
        help="Number of slowest relaxation times to record.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="robustness_vs_deltaE.csv",
        help="Output CSV filename (written in data-dir).",
    )

    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    deltaE_values = [float(x.strip()) for x in args.deltaE_grid.split(",") if x.strip()]

    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] ΔE_cut grid: {deltaE_values}")
    print(f"[INFO] E_window: {args.E_window}, T={args.temperature} K")

    # Read A/B sets (0-based)
    A_ids = read_min_list(data_dir / "min.A")
    B_ids = read_min_list(data_dir / "min.B")
    if A_ids.size == 0 or B_ids.size == 0:
        print("[WARN] min.A or min.B not found or empty; MFPTs will not be computed.")

    rows: list[Dict[str, Any]] = []

    for dE in deltaE_values:
        print(f"\n[INFO] === ΔE_cut = {dE:.3f} ===")
        try:
            Q_eff, kept_order, keep_ids = build_Qeff_for_deltaE(
                dps_dir=data_dir,
                deltaE_cut=dE,
                E_window=args.E_window,
                temperature=args.temperature,
            )
        except NotImplementedError as e:
            print("[ERROR] build_Qeff_for_deltaE is not implemented yet.")
            print("        Fill in your KTN + GT code in that function.")
            return

        # Q_eff assumed row generator
        n_eff = Q_eff.shape[0]
        print(f"[INFO] Size of coarse-grained generator: {n_eff} states")

        # Map A/B (original minima indices) into coarse-grained indices
        # kept_order is an array of original 0-based indices in the order of Q_eff rows
        A_eff = np.intersect1d(A_ids, kept_order, assume_unique=False)
        B_eff = np.intersect1d(B_ids, kept_order, assume_unique=False)

        # Convert to positions in Q_eff (0..n_eff-1)
        # kept_order[pos] = original_index
        inv_map = {orig: pos for pos, orig in enumerate(kept_order)}
        A_pos = np.array([inv_map[i] for i in A_eff], dtype=int) if A_eff.size > 0 else np.array([], dtype=int)
        B_pos = np.array([inv_map[i] for i in B_eff], dtype=int) if B_eff.size > 0 else np.array([], dtype=int)

        if A_pos.size > 0 and B_pos.size > 0:
            mfpt_AB = compute_mfpt_from_Q(Q_eff, A_pos, B_pos)
            mfpt_BA = compute_mfpt_from_Q(Q_eff, B_pos, A_pos)
        else:
            mfpt_AB = np.nan
            mfpt_BA = np.nan
            print("[WARN] Could not map A/B sets into coarse-grained states; MFPTs set to NaN.")

        # Relaxation times
        t_relax = leading_relaxation_times(Q_eff, k=args.n_relax)
        # Pad to fixed length for CSV
        t_pad = np.full(args.n_relax, np.nan)
        t_pad[: min(args.n_relax, len(t_relax))] = t_relax[: args.n_relax]

        row: Dict[str, Any] = {
            "deltaE_cut": dE,
            "N_eff": n_eff,
            "MFPT_A_to_B": mfpt_AB,
            "MFPT_B_to_A": mfpt_BA,
        }
        for k_idx in range(args.n_relax):
            row[f"t_relax_{k_idx+1}"] = t_pad[k_idx]

        rows.append(row)

    # Write CSV
    out_path = data_dir / args.out_csv
    # Manual CSV write to avoid pandas dependency if you prefer
    if rows:
        keys = list(rows[0].keys())
        with out_path.open("w") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"\n[OK] Wrote robustness summary to {out_path}")
    else:
        print("[WARN] No rows written; something went wrong.")


if __name__ == "__main__":
    main()
