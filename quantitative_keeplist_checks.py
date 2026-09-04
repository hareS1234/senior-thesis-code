#!/usr/bin/env python
"""Check how sensitive the coarse kinetics are to the basin cutoff.

The project-specific Q_eff builder is still a stub below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Dict, Any, Tuple

import numpy as np

from generate_basin_keep_lists import build_basin_keep_set
from ktn_utils import compute_mfpt_from_Q, leading_relaxation_times





def read_min_list(path: Path) -> np.ndarray:
    """Read min.A/min.B and convert the IDs to zero-based indices."""
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
                ids.append(mid - 1)
            except ValueError:
                continue
    return np.asarray(ids, dtype=int)





def build_Qeff_for_deltaE(
    dps_dir: Path,
    deltaE_cut: float,
    E_window: float,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hook for building Q_eff at one cutoff.

    This is still a stub; wire in the project's PyGT calls before using it.
    """


    keep_ids_1based = build_basin_keep_set(
        data_dir=dps_dir,
        deltaE_cut=deltaE_cut,
        E_window=E_window,
    )

    keep_ids = np.array(keep_ids_1based, dtype=int) - 1














    raise NotImplementedError(
        "You need to hook in your existing KTN + GT code in build_Qeff_for_deltaE."
    )








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


        n_eff = Q_eff.shape[0]
        print(f"[INFO] Size of coarse-grained generator: {n_eff} states")



        A_eff = np.intersect1d(A_ids, kept_order, assume_unique=False)
        B_eff = np.intersect1d(B_ids, kept_order, assume_unique=False)



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


        t_relax = leading_relaxation_times(Q_eff, k=args.n_relax)

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


    out_path = data_dir / args.out_csv

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
