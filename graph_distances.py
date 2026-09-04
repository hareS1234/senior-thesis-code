"""Distance helpers for the retained-minima graph."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

from config import MarkovFilePaths
from stationary_point_io import build_barrier_matrix


def load_sparse(path: Path) -> csr_matrix:
    from scipy.sparse import load_npz

    mat = load_npz(path)
    if not isinstance(mat, csr_matrix):
        mat = mat.tocsr()
    return mat


def barrier_distances(
    data_dir: Path,
    markov_paths: MarkovFilePaths,
    sources: Optional[Iterable[int]] = None,
) -> np.ndarray:
    """Run Dijkstra on the barrier graph.

    With no ``sources`` this returns the full N-by-N matrix; otherwise it only
    keeps the requested source rows.
    """
    barrier_mat = build_barrier_matrix(data_dir, markov_paths)
    N = barrier_mat.shape[0]

    if sources is None:
        indices = None
    else:
        indices = np.array(list(sources), dtype=int)
        assert indices.ndim == 1
        if np.any((indices < 0) | (indices >= N)):
            raise ValueError("Some source indices are out of range")

    dist = shortest_path(
        barrier_mat,
        directed=False,
        indices=indices,
        unweighted=False,
    )
    return np.asarray(dist)


def rate_based_lengths(
    markov_paths: MarkovFilePaths,
    sources: Optional[Iterable[int]] = None,
    min_rate: float = 1e-300,
) -> np.ndarray:
    """Run Dijkstra with ``-log(k_ij)`` edge lengths.

    Rates below ``min_rate`` are ignored. ``sources`` works the same way as in
    ``barrier_distances``.
    """
    K = load_sparse(markov_paths.K_path)
    N = K.shape[0]



    K_coo = K.tocoo()
    mask = (K_coo.row != K_coo.col) & (K_coo.data > min_rate)
    src = K_coo.col[mask]
    dst = K_coo.row[mask]
    data = -np.log(K_coo.data[mask])


    if data.size and np.min(data) < 0:
        data = data - np.min(data)

    from scipy.sparse import coo_matrix

    L = coo_matrix((data, (src, dst)), shape=(N, N)).tocsr()

    if sources is None:
        indices = None
    else:
        indices = np.array(list(sources), dtype=int)
        assert indices.ndim == 1
        if np.any((indices < 0) | (indices >= N)):
            raise ValueError("Some source indices are out of range")

    dist = shortest_path(
        L,
        directed=True,
        indices=indices,
        unweighted=False,
    )
    return np.asarray(dist)
