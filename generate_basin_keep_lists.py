#!/usr/bin/env python
"""
generate_basin_keep_lists.py

Basin-based keep-list generation for PATHSAMPLE DPS directories.

- Walks a root directory recursively.
- For every folder containing both min.data and ts.data,
  it builds a basin-based keep set using a barrier-height
  threshold ΔE_cut and writes:

  keep_minima_dE{ΔE_cut}.txt
  keep_minima.txt      (copy of the same list, for compatibility)

Basin definition:
- Nodes: minima.
- Edge i--j exists if there is at least one TS with
      ΔE_ij = E_TS - min(E_i, E_j) <= ΔE_cut.
- Basins = connected components of this graph.

Kept minima K:
- All minima listed in min.A and min.B (if present),
- The lowest-energy minimum in each connected component,
- The global minimum,
- Optionally, any minimum with E - Emin <= E_window.

All indices are original PATHSAMPLE minimum IDs (1-based).
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


# ---------- Basic readers ----------

def read_min_energies(min_data_path: Path) -> np.ndarray:
    """Read energies (first column) from min.data."""
    if not min_data_path.exists():
        raise FileNotFoundError(f"{min_data_path} not found")
    E = np.loadtxt(min_data_path, usecols=0)
    return np.asarray(E, dtype=float)


def read_min_list(path: Path) -> List[int]:
    """Read a PATHSAMPLE list file (min.A or min.B), first int per line."""
    if not path.exists():
        return []
    ids: List[int] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                ids.append(int(parts[0]))
            except ValueError:
                continue
    return ids


def read_ts_file(ts_path: Path) -> List[Tuple[float, int, int]]:
    """
    Read ts.data and return a list of (E_ts, min1, min2).

    NOTE: This assumes the standard PATHSAMPLE format where:
        column 1 = TS energy
        column 4 = index of first minimum
        column 5 = index of second minimum

    If your ts.data format differs, adjust the column indices below.
    """
    if not ts_path.exists():
        raise FileNotFoundError(f"{ts_path} not found")

    ts_records: List[Tuple[float, int, int]] = []

    with ts_path.open() as fh:
        for line in fh:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = line.split()
            try:
                E_ts = float(parts[0])
                # standard: min1 = col 3 or 4, min2 = col 4 or 5.
                # For Nicy / Wales ts.data, min indices are in cols 3 and 4 (0-based 3,4)
                # but sometimes 4 and 5. Adjust if needed.
                min1 = int(parts[3])
                min2 = int(parts[4])
            except (ValueError, IndexError):
                continue
            ts_records.append((E_ts, min1, min2))

    return ts_records


# ---------- Basin construction ----------

def build_low_barrier_graph(
    E: np.ndarray,
    ts_records: List[Tuple[float, int, int]],
    deltaE_cut: float,
) -> Dict[int, Set[int]]:
    """
    Build an undirected adjacency list on minima using barrier threshold ΔE_cut.

    Nodes: minima indices (1-based).
    Edge i--j if there exists at least one TS with
        ΔE_ij = E_TS - min(E_i, E_j) <= deltaE_cut.
    """
    n_min = E.shape[0]
    adj: Dict[int, Set[int]] = {i: set() for i in range(1, n_min + 1)}

    for E_ts, min1, min2 in ts_records:
        if min1 < 1 or min2 < 1 or min1 > n_min or min2 > n_min:
            continue
        Emin_pair = min(E[min1 - 1], E[min2 - 1])
        dE = E_ts - Emin_pair
        if dE <= deltaE_cut:
            adj[min1].add(min2)
            adj[min2].add(min1)

    return adj


def connected_components(adj: Dict[int, Set[int]]) -> List[List[int]]:
    """
    Return list of connected components (as lists of node IDs)
    for an undirected graph given by adjacency list 'adj'.
    """
    visited: Set[int] = set()
    comps: List[List[int]] = []

    for start in adj.keys():
        if start in visited:
            continue
        queue = deque([start])
        comp: List[int] = []
        visited.add(start)

        while queue:
            u = queue.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        comps.append(comp)

    return comps


def build_basin_keep_set(
    data_dir: Path,
    deltaE_cut: float,
    E_window: float | None = None,
) -> List[int]:
    """
    Construct a basin-based keep set K for a single DPS directory.

    Parameters
    ----------
    data_dir : Path
        Directory containing min.data, ts.data, and optionally min.A/min.B.
    deltaE_cut : float
        Barrier threshold. Edge i--j if ΔE_ij <= deltaE_cut.
    E_window : float or None
        If not None, also include all minima with E - Emin <= E_window.

    Returns
    -------
    keep_ids : sorted list of unique 1-based minima indices.
    """
    min_data_path = data_dir / "min.data"
    ts_data_path = data_dir / "ts.data"

    E = read_min_energies(min_data_path)
    n_min = E.shape[0]

    # Global minimum (1-based)
    global_id = int(np.argmin(E) + 1)
    Emin = float(E[global_id - 1])

    # Optional A/B sets
    A_ids = read_min_list(data_dir / "min.A")
    B_ids = read_min_list(data_dir / "min.B")
    AB_ids: Set[int] = set(A_ids) | set(B_ids)

    # Transition states and low-barrier graph
    ts_records = read_ts_file(ts_data_path)
    adj = build_low_barrier_graph(E, ts_records, deltaE_cut)

    # Connected components = basins
    comps = connected_components(adj)

    keep: Set[int] = set()
    keep.add(global_id)     # just to be explicit

    # Within each component, keep AB minima and lowest-energy rep
    for comp in comps:
        comp_array = np.array(comp, dtype=int)
        comp_E = E[comp_array - 1]
        rep_local_idx = int(np.argmin(comp_E))
        rep_min = int(comp_array[rep_local_idx])

        keep.add(rep_min)
        for mid in comp:
            if mid in AB_ids:
                keep.add(mid)

    # Optional energy window around global minimum
    if E_window is not None:
        for idx, Ei in enumerate(E, start=1):
            if Ei - Emin <= E_window:
                keep.add(idx)

    keep_ids = sorted(keep)
    return keep_ids


# ---------- IO helpers ----------

def write_keep_files(
    data_dir: Path,
    keep_ids: List[int],
    deltaE_cut: float,
    overwrite: bool = False,
) -> None:
    """
    Write both:
      - keep_minima_dE{deltaE_cut}.txt
      - keep_minima.txt              (copy, for PyGT/PATHSAMPLE)
    in data_dir.
    """
    label = f"dE{deltaE_cut:.2f}".replace(".", "p")
    fname_labelled = f"keep_minima_{label}.txt"
    path_labelled = data_dir / fname_labelled
    path_default = data_dir / "keep_minima.txt"

    if path_labelled.exists() and not overwrite:
        print(f"[SKIP] {path_labelled} exists (use --overwrite to replace).")
    else:
        with path_labelled.open("w") as fh:
            for mid in keep_ids:
                fh.write(f"{mid:d}\n")
        print(f"[OK]   wrote {len(keep_ids)} minima to {path_labelled}")

    # Also update keep_minima.txt for compatibility
    if (not path_default.exists()) or overwrite:
        with path_default.open("w") as fh:
            for mid in keep_ids:
                fh.write(f"{mid:d}\n")
        print(f"[OK]   wrote default keep_minima.txt in {data_dir}")
    else:
        print(f"[SKIP] keep_minima.txt exists and --overwrite not set.")


# ---------- Driver over the full tree ----------

def find_dps_dirs(root: Path) -> List[Path]:
    """
    Find all directories under 'root' that contain both min.data and ts.data.
    These are treated as DPS directories (monomer/dimer, 19sb/99idps, etc.).
    """
    dps_dirs: List[Path] = []
    for min_file in root.rglob("min.data"):
        data_dir = min_file.parent
        ts_file = data_dir / "ts.data"
        if ts_file.exists():
            dps_dirs.append(data_dir)
    return sorted(dps_dirs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate basin-based keep_minima.txt files using a barrier threshold ΔE_cut."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        required=True,
        help="Root directory to search (e.g. /scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped).",
    )
    parser.add_argument(
        "--deltaE",
        type=float,
        required=True,
        help="Barrier threshold ΔE_cut (same energy units as min.data / ts.data).",
    )
    parser.add_argument(
        "--E-window",
        type=float,
        default=None,
        help="Optional energy window E - Emin <= E_window to include low-lying minima.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing keep_minima*.txt files.",
    )
    args = parser.parse_args()

    root = args.root_dir.resolve()
    deltaE_cut = args.deltaE

    dps_dirs = find_dps_dirs(root)
    if not dps_dirs:
        print(f"No DPS directories with min.data and ts.data found under {root}")
        return

    print(f"Found {len(dps_dirs)} DPS directories under {root}")
    print(f"Using ΔE_cut = {deltaE_cut:.3f}, E_window = {args.E_window}")

    for d in dps_dirs:
        print(f"\n[DIR] {d}")
        keep_ids = build_basin_keep_set(d, deltaE_cut=deltaE_cut, E_window=args.E_window)
        write_keep_files(d, keep_ids, deltaE_cut=deltaE_cut, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
