#!/usr/bin/env python
"""
qualitative_keeplist_checks.py

Qualitative sanity checks for basin-based keep lists:

1) Plot the distribution of single-edge barrier heights from ts.data
2) For a grid of ΔE_cut values, compute the number of kept minima in the
   basin-based scheme and plot N_keep vs ΔE_cut.

Requires:
- min.data with energies in column 0
- ts.data with TS energy in column 0 and endpoint minima indices (1-based)
  in columns 1 and 2
- generate_basin_keep_lists.py exposing build_basin_keep_set(...)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

# adjust this import if your function lives elsewhere / has a different name
from generate_basin_keep_lists import build_basin_keep_set


def load_min_energies(min_data_path: Path) -> np.ndarray:
    """Load minimum energies from min.data (assumes energy in column 0)."""
    if not min_data_path.exists():
        raise FileNotFoundError(f"{min_data_path} not found")
    E = np.loadtxt(min_data_path, usecols=0)
    return np.asarray(E, dtype=float)


def load_ts_data(ts_data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load TS energies and endpoint minima indices from ts.data.

    Assumes:
      col 0: TS energy
      col 1: first minimum index (1-based)
      col 2: second minimum index (1-based)
    """
    if not ts_data_path.exists():
        raise FileNotFoundError(f"{ts_data_path} not found")

    arr = np.loadtxt(ts_data_path, usecols=(0, 1, 2))
    E_ts = np.asarray(arr[:, 0], dtype=float)
    m1 = np.asarray(arr[:, 1], dtype=int)
    m2 = np.asarray(arr[:, 2], dtype=int)
    return E_ts, m1, m2


def compute_single_edge_barriers(
    E_min: np.ndarray,
    E_ts: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
) -> np.ndarray:
    """
    For each TS, compute barrier height above the lower of the two minima:

      ΔE_edge = E_TS - min(E_i, E_j)

    Returns an array of ΔE_edge (same length as number of TSs).
    """
    # indices in files are 1-based; convert to 0-based for numpy
    E1 = E_min[m1 - 1]
    E2 = E_min[m2 - 1]
    Emin_pair = np.minimum(E1, E2)
    dE = E_ts - Emin_pair
    return dE


def parse_deltaE_list(raw: str) -> Sequence[float]:
    """
    Parse a comma-separated list like '10,20,40' into [10.0, 20.0, 40.0].
    """
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Qualitative sanity checks for basin-based keep lists."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="DPS directory containing min.data, ts.data, etc.",
    )
    parser.add_argument(
        "--deltaE-grid",
        type=str,
        default="10,20,30,40,50,60",
        help="Comma-separated list of ΔE_cut values to test (same units as min.data). "
             "Example: '5,10,20,40,80'.",
    )
    parser.add_argument(
        "--E-window",
        type=float,
        default=3.0,
        help="Energy window (E - Emin <= E_window) used in the basin keep-list builder.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for the barrier-height histogram.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="qualitative_checks",
        help="Prefix for output PNG filenames.",
    )
    parser.add_argument(
        "--highlight-deltaE",
        type=float,
        default=None,
        help="Optional ΔE_cut to highlight as a vertical line in the histogram.",
    )

    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    deltaE_values = parse_deltaE_list(args.deltaE_grid)

    min_data_path = data_dir / "min.data"
    ts_data_path = data_dir / "ts.data"

    print(f"[INFO] Using data directory: {data_dir}")
    print(f"[INFO] Loading energies from: {min_data_path}")
    print(f"[INFO] Loading TS data from: {ts_data_path}")

    E_min = load_min_energies(min_data_path)
    E_ts, m1, m2 = load_ts_data(ts_data_path)

    # ------------------------------
    # 1) Barrier height distribution
    # ------------------------------
    dE_edges = compute_single_edge_barriers(E_min, E_ts, m1, m2)

    print(f"[INFO] Number of TS edges: {len(dE_edges)}")
    print(f"[INFO] Barrier stats (E_TS - min(E_i,E_j)):")
    print(f"       min  = {np.min(dE_edges):.3f}")
    print(f"       max  = {np.max(dE_edges):.3f}")
    print(f"       mean = {np.mean(dE_edges):.3f}")
    print(f"       25%  = {np.percentile(dE_edges, 25):.3f}")
    print(f"       50%  = {np.percentile(dE_edges, 50):.3f}")
    print(f"       75%  = {np.percentile(dE_edges, 75):.3f}")
    print(f"       90%  = {np.percentile(dE_edges, 90):.3f}")

    plt.figure(figsize=(6, 4))
    plt.hist(dE_edges, bins=args.hist_bins, density=True, alpha=0.7)
    plt.xlabel(r"Barrier height $\Delta E_{\mathrm{edge}}$ (TS - lower minimum)")
    plt.ylabel("Probability density")
    plt.title("Distribution of single-edge barrier heights")

    if args.highlight_deltaE is not None:
        plt.axvline(
            args.highlight_deltaE,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=fr"$\Delta E_{{\mathrm{{cut}}}}={args.highlight_deltaE:.1f}$",
        )
        plt.legend()

    hist_path = data_dir / f"{args.out_prefix}_barrier_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"[OK]  Saved barrier histogram to {hist_path}")

    # ---------------------------------------------
    # 2) N_keep vs ΔE_cut using basin-based scheme
    # ---------------------------------------------
    N_keep_list = []
    for dE in deltaE_values:
        keep_ids = build_basin_keep_set(
            data_dir=data_dir,
            deltaE_cut=dE,
            E_window=args.E_window,
        )
        N_keep_list.append(len(keep_ids))
        print(f"[INFO] ΔE_cut={dE:.1f} -> N_keep={len(keep_ids)}")

    plt.figure(figsize=(6, 4))
    plt.plot(deltaE_values, N_keep_list, marker="o")
    plt.xlabel(r"$\Delta E_{\mathrm{cut}}$ (same units as energies)")
    plt.ylabel(r"$N_{\mathrm{keep}}$")
    plt.title("Number of kept minima vs $\Delta E_{\mathrm{cut}}$")
    plt.grid(True, alpha=0.3)

    nkeep_path = data_dir / f"{args.out_prefix}_Nkeep_vs_deltaE.png"
    plt.tight_layout()
    plt.savefig(nkeep_path, dpi=200)
    plt.close()
    print(f"[OK]  Saved N_keep vs ΔE plot to {nkeep_path}")


if __name__ == "__main__":
    main()
