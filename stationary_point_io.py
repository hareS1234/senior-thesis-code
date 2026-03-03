"""
stationary_point_io.py

Low-level I/O for PATHSAMPLE / KTN stationary-point data.

- Read min.data and ts.data
- Build a barrier-height graph aligned with the retained minima
  used by PyGT (largest connected component).

Barrier height for edge i--j is defined as:
    barrier(i,j) = min_over_TS [ E_TS - min(E_i, E_j) ]
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from scipy.sparse import coo_matrix, csr_matrix

from config import MarkovFilePaths


def read_min_ts(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read min.data and ts.data.

    Returns
    -------
    min_energies : (n_min,) array
        Energies of minima in *original* PATHSAMPLE indexing (1-based in files -> 0-based here).
    ts_energies : (n_ts,) array
        Energies of transition states.
    ts_conn : (n_ts, 2) int array
        Connected minima indices in original 1-based indexing (copied directly).
    """
    min_path = data_dir / "min.data"
    ts_path = data_dir / "ts.data"
    if not min_path.exists() or not ts_path.exists():
        raise FileNotFoundError(f"Expected min.data and ts.data in {data_dir}")

    # min.data: E S DD RX RY RZ
    min_data = np.loadtxt(
        min_path,
        dtype={
            "names": ("E", "S", "DD", "RX", "RY", "RZ"),
            "formats": (float, float, int, float, float, float),
        },
    )
    min_energies = np.asarray(min_data["E"], dtype=float)

    # ts.data: E S DD F I RX RY RZ  (F/I are minima IDs)
    ts_data = np.loadtxt(
        ts_path,
        dtype={
            "names": ("E", "S", "DD", "F", "I", "RX", "RY", "RZ"),
            "formats": (float, float, int, int, int, float, float, float),
        },
    )
    ts_energies = np.asarray(ts_data["E"], dtype=float)
    ts_conn = np.vstack([ts_data["F"], ts_data["I"]]).T  # shape (n_ts, 2), 1-based IDs

    return min_energies, ts_energies, ts_conn


def build_barrier_matrix(
    data_dir: Path,
    markov_paths: MarkovFilePaths,
    overwrite: bool = False,
) -> csr_matrix:
    """
    Construct a sparse symmetric matrix of barrier heights between
    retained minima and save to disk.

    Parameters
    ----------
    data_dir : Path
        Directory containing min.data / ts.data.
    markov_paths : MarkovFilePaths
        Gives temperature-specific pygt_dir and retained_mask path.
        NOTE: barrier heights are temperature-independent; we just
        use the same retained-mask mapping as PyGT.
    overwrite : bool
        If False and barrier_matrix.npz exists, just load and return.

    Returns
    -------
    barrier_mat : csr_matrix (N, N)
        Symmetric matrix with barrier_mat[i, j] >= 0, stored in CSR.
        N = number of retained minima.
    """
    out_path = markov_paths.barrier_matrix_path
    if out_path.exists() and not overwrite:
        from scipy.sparse import load_npz

        return load_npz(out_path)

    retained_mask = np.load(markov_paths.retained_mask_path)  # length = nnodes in min.data
    original_ids = np.nonzero(retained_mask)[0] + 1  # original 1-based IDs retained
    N = retained_mask.sum()

    # Map original (1-based) min IDs -> local [0..N-1] index
    orig_to_local: Dict[int, int] = {
        int(orig_id): idx for idx, orig_id in enumerate(original_ids)
    }

    min_E, ts_E, ts_conn = read_min_ts(data_dir)

    # Accumulate smallest barrier for each undirected edge
    edge_barriers: Dict[tuple[int, int], float] = {}

    for E_ts, (F, I) in zip(ts_E, ts_conn):
        F = int(F)
        I = int(I)
        if F == I:
            continue

        if F not in orig_to_local or I not in orig_to_local:
            # skip TS involving a pruned minimum
            continue

        i_local = orig_to_local[F]
        j_local = orig_to_local[I]
        if i_local == j_local:
            continue

        E_i = float(min_E[F - 1])
        E_j = float(min_E[I - 1])
        barrier = float(E_ts - min(E_i, E_j))
        if barrier < 0.0:
            # Numerical noise; clamp
            barrier = 0.0

        # Undirected edge key
        a, b = sorted((i_local, j_local))
        key = (a, b)
        if key not in edge_barriers or barrier < edge_barriers[key]:
            edge_barriers[key] = barrier

    if not edge_barriers:
        raise RuntimeError(f"No barriers found for retained minima in {data_dir}")

    rows = []
    cols = []
    data = []
    for (i, j), b in edge_barriers.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([b, b])

    barrier_mat = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    barrier_mat.eliminate_zeros()

    markov_paths.pygt_dir.mkdir(parents=True, exist_ok=True)
    from scipy.sparse import save_npz

    save_npz(out_path, barrier_mat)
    return barrier_mat
