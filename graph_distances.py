"""
graph_distances.py

Higher-level helpers to compute graph distances on the KTN:

- barrier-based Dijkstra distance (additive sum of barrier heights)
- rate-based Dijkstra distance (edge length = -log k_ij)

All distances are on the retained-minima graph (same indexing as B/K/Q/pi).
"""

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
    """
    Compute Dijkstra distances on the barrier-height graph.

    Parameters
    ----------
    data_dir : Path
        DPS directory containing min.data / ts.data.
    markov_paths : MarkovFilePaths
        Gives pygt_dir + retained_mask path.
    sources : iterable of int or None
        Indices (in retained-minima indexing) to use as sources.
        If None, distances from *all* nodes are computed (N×N).

    Returns
    -------
    dist : ndarray
        If sources is None: shape (N, N).
        Else: shape (len(sources), N).
        dist[i, j] = minimal sum of barrier heights along any path i→j.
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
    """
    Compute Dijkstra distances where edge length = -log(k_ij).

    Uses the off-diagonal rate matrix K from PyGT.

    Parameters
    ----------
    markov_paths : MarkovFilePaths
        Path bundle pointing to K_TxxxK.npz.
    sources : iterable of int or None
        As in barrier_distances.
    min_rate : float
        Smallest rate to allow (values below are ignored).

    Returns
    -------
    dist : ndarray
        Dijkstra distances on -log(k_ij).
    """
    K = load_sparse(markov_paths.K_path)
    N = K.shape[0]

    # Build sparse adjacency lengths in (src, dst) orientation.
    # K[dst, src] = k_{dst <- src} so each nonzero K[i, j] is edge j -> i.
    K_coo = K.tocoo()
    mask = (K_coo.row != K_coo.col) & (K_coo.data > min_rate)
    src = K_coo.col[mask]
    dst = K_coo.row[mask]
    data = -np.log(K_coo.data[mask])

    # Keep edge lengths non-negative for Dijkstra compatibility.
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
        directed=True,  # rate graph is directed
        indices=indices,
        unweighted=False,
    )
    return np.asarray(dist)
