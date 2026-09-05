#!/usr/bin/env python
"""Check how sensitive the coarse kinetics are to the basin cutoff.

Each cutoff produces a basin keep set, a PyGT-reduced generator, endpoint
MFPTs, and a few slow relaxation times.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import scipy.sparse as sp

from generate_basin_keep_lists import build_basin_keep_set, read_min_list as read_min_ids
from ktn_utils import compute_mfpt_from_Q, leading_relaxation_times





def read_min_list(path: Path) -> np.ndarray:
    """Read min.A/min.B and convert the IDs to zero-based indices."""
    return np.asarray(read_min_ids(path), dtype=int) - 1





def build_Qeff_for_deltaE(
    dps_dir: Path,
    deltaE_cut: float,
    E_window: float,
    temperature: float,
    block: int = 50,
    cond_thresh: float = 1e13,
    screen: bool = False,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
    """Build one PyGT-reduced generator from a basin keep set."""
    keep_ids_1based = build_basin_keep_set(
        data_dir=dps_dir,
        deltaE_cut=deltaE_cut,
        E_window=E_window,
    )
    requested_ids = np.asarray(keep_ids_1based, dtype=int)

    tag = f"T{int(round(temperature))}K"
    markov_dir = dps_dir / f"markov_{tag}"
    B_path = markov_dir / f"B_{tag}.npz"
    tau_path = markov_dir / f"tau_{tag}.npy"
    orig_path = markov_dir / f"original_min_ids_{tag}.npy"
    if not orig_path.exists():
        orig_path = markov_dir / f"orig_min_ids_{tag}.npy"

    missing_files = [p for p in (B_path, tau_path, orig_path) if not p.exists()]
    if missing_files:
        missing_text = ", ".join(str(p) for p in missing_files)
        raise FileNotFoundError(f"Microscopic model files are missing: {missing_text}")

    B = sp.load_npz(B_path).tocsr()
    tau = np.asarray(np.load(tau_path), dtype=float).ravel()
    original_ids = np.asarray(np.load(orig_path), dtype=int).ravel()

    if B.shape[0] != B.shape[1] or B.shape[0] != tau.size:
        raise ValueError("B and tau have incompatible shapes.")
    if original_ids.size != tau.size:
        raise ValueError("original_min_ids and tau have incompatible lengths.")
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError("tau must contain finite, positive waiting times.")

    keep_mask = np.isin(original_ids, requested_ids)
    if int(keep_mask.sum()) < 2:
        raise ValueError("The keep set has fewer than two retained microscopic states.")

    missing_ids = np.setdiff1d(requested_ids, original_ids)
    if missing_ids.size:
        preview = ", ".join(str(i) for i in missing_ids[:8])
        suffix = " ..." if missing_ids.size > 8 else ""
        print(
            f"[WARN] {missing_ids.size} requested minima are outside the retained "
            f"microscopic component: {preview}{suffix}"
        )

    remove_mask = ~keep_mask
    if remove_mask.any():
        try:
            import PyGT.GT as GT
        except ImportError as exc:
            raise ImportError(
                "PyGT is required for the quantitative keep-list check. "
                "Install the pinned requirements first."
            ) from exc

        _, _, Q_eff = GT.blockGT(
            rm_vec=remove_mask,
            B=B,
            tau=tau,
            block=int(block),
            order=None,
            rates=True,
            screen=bool(screen),
            cond_thresh=float(cond_thresh),
        )
        Q_eff = sp.csr_matrix(Q_eff, dtype=float)
    else:
        escape = 1.0 / tau
        Q_eff = (B @ sp.diags(escape) - sp.diags(escape)).tocsr()

    kept_order = original_ids[keep_mask] - 1
    keep_ids = requested_ids - 1
    if Q_eff.shape != (kept_order.size, kept_order.size):
        raise RuntimeError("PyGT output order does not match the retained keep set.")
    return Q_eff, kept_order, keep_ids








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
        help="Temperature in K.",
    )
    parser.add_argument(
        "--n-relax",
        type=int,
        default=5,
        help="Number of slowest relaxation times to record.",
    )
    parser.add_argument(
        "--block",
        type=int,
        default=50,
        help="Maximum PyGT block size.",
    )
    parser.add_argument(
        "--cond-thresh",
        type=float,
        default=1e13,
        help="PyGT block condition threshold.",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Print PyGT reduction progress.",
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
    if not deltaE_values:
        parser.error("--deltaE-grid must contain at least one number.")

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
        Q_eff, kept_order, keep_ids = build_Qeff_for_deltaE(
            dps_dir=data_dir,
            deltaE_cut=dE,
            E_window=args.E_window,
            temperature=args.temperature,
            block=args.block,
            cond_thresh=args.cond_thresh,
            screen=args.screen,
        )


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
            "N_requested": int(keep_ids.size),
            "N_eff": n_eff,
            "N_dropped_outside_micro": int(keep_ids.size - kept_order.size),
            "MFPT_A_to_B": mfpt_AB,
            "MFPT_B_to_A": mfpt_BA,
        }
        for k_idx in range(args.n_relax):
            row[f"t_relax_{k_idx+1}"] = t_pad[k_idx]

        rows.append(row)


    out_path = data_dir / args.out_csv

    if rows:
        keys = list(rows[0].keys())
        with out_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[OK] Wrote robustness summary to {out_path}")
    else:
        print("[WARN] No rows written; something went wrong.")


if __name__ == "__main__":
    main()
