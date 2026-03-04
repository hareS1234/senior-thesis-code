#!/usr/bin/env python
"""
graph_features.py

Compute graph-theoretic features for each coarse-grained KTN at a given
temperature.  Outputs a single CSV with one row per network and ~50
structural features spanning distance, spectral, centrality, community,
path, and topology categories.

Usage (on the cluster):
    python graph_features.py --out graph_features_coarse_T300K.csv
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, coo_matrix
from scipy.sparse.csgraph import shortest_path, connected_components
from scipy.sparse.linalg import eigsh, ArpackNoConvergence, eigs
from scipy.stats import skew as scipy_skew

from config import BASE_DIR, TEMPERATURES, MarkovFilePaths, iter_dps_dirs
from io_markov import load_markov, load_AB_selectors, temp_tag


# ======================================================================
#  1. Distance features
# ======================================================================

def _branching_length_matrix(B: csr_matrix) -> csr_matrix:
    """Build a sparse *adjacency/length* matrix for Dijkstra on the branching graph.

    Conventions
    ----------
    The branching matrix uses the same convention as the rates:
        B[i, j] = P(jump to i | leaving j) = B_{i <- j}

    SciPy's `shortest_path` expects an adjacency matrix A where A[src, dst] is
    the edge weight from `src` to `dst`. Therefore we must *transpose* the
    (i <- j) storage convention:

        L[src=j, dst=i] = -log(B[i, j])

    This yields non-negative edge weights suitable for Dijkstra's algorithm.
    """
    B_coo = B.tocoo()
    mask = B_coo.data > 0
    # Stored as (dst=i, src=j) -> convert to adjacency (src=j, dst=i)
    src = B_coo.col[mask]
    dst = B_coo.row[mask]
    data = -np.log(np.clip(B_coo.data[mask], 1e-300, None))
    return coo_matrix((data, (src, dst)), shape=B.shape).tocsr()


def _rate_length_matrix(K: csr_matrix, min_rate: float = 1e-300) -> csr_matrix:
    """Build a sparse *adjacency/length* matrix for shortest paths on the rate graph.

    Conventions
    ----------
    K[i, j] = k_{i <- j} is the rate *into i from j* (so columns are sources).

    For a directed edge j -> i, we want the adjacency entry:
        L[src=j, dst=i] = -log(k_{i <- j}) = -log(K[i, j])

    Non-negativity
    --------------
    Dijkstra's algorithm (used internally by SciPy when possible) requires
    non-negative edge weights. In principle, rates can exceed 1 in the chosen
    units, which would make -log(k) negative and can create negative cycles.

    To make this descriptor robust, we *shift* the edge lengths so that the
    minimum edge length is 0 whenever needed. This is equivalent to normalizing
    rates by the maximum observed rate in the graph, and leaves the ordering of
    edges by "fastness" intact.
    """
    K_coo = K.tocoo()
    mask = (K_coo.row != K_coo.col) & (K_coo.data > min_rate)
    # Stored as (dst=i, src=j) -> adjacency (src=j, dst=i)
    src = K_coo.col[mask]
    dst = K_coo.row[mask]
    data = -np.log(np.clip(K_coo.data[mask], min_rate, None))

    # Shift to non-negative if needed (safety guard)
    if data.size and np.min(data) < 0:
        data = data - np.min(data)

    return coo_matrix((data, (src, dst)), shape=K.shape).tocsr()


def compute_distance_features(
    B: csr_matrix,
    A_sel: np.ndarray,
    B_sel: np.ndarray,
    barrier_mat: Optional[csr_matrix] = None,
) -> Dict[str, float]:
    """
    Compute A<->B distance features from branching and barrier matrices.

    Branching-based distances use edge weight = -log(B_ij) where B is
    the branching probability matrix (entries in (0,1]), computed via
    Dijkstra on a directed graph.  Barrier-based distances use the
    undirected barrier-height matrix (if provided).
    """
    feats: Dict[str, float] = {}
    A_idx = np.where(A_sel)[0]
    B_idx_dist = np.where(B_sel)[0]

    # --- Branching-probability distances (directed, non-negative) ---
    L = _branching_length_matrix(B)
    # A -> B
    dist_from_A = shortest_path(L, directed=True, indices=A_idx)
    ab_dists = dist_from_A[:, B_idx_dist]  # |A| x |B|
    finite_ab = ab_dists[np.isfinite(ab_dists)]
    feats["rate_dist_AB_min"] = float(np.min(finite_ab)) if finite_ab.size else np.nan
    feats["rate_dist_AB_mean"] = float(np.mean(finite_ab)) if finite_ab.size else np.nan

    # B -> A
    dist_from_B = shortest_path(L, directed=True, indices=B_idx_dist)
    ba_dists = dist_from_B[:, A_idx]
    finite_ba = ba_dists[np.isfinite(ba_dists)]
    feats["rate_dist_BA_min"] = float(np.min(finite_ba)) if finite_ba.size else np.nan
    feats["rate_dist_BA_mean"] = float(np.mean(finite_ba)) if finite_ba.size else np.nan

    # Asymmetry
    if finite_ab.size and finite_ba.size:
        feats["rate_dist_asymmetry"] = abs(feats["rate_dist_AB_min"] - feats["rate_dist_BA_min"])
    else:
        feats["rate_dist_asymmetry"] = np.nan

    # --- Barrier-based distances (undirected) ---
    if barrier_mat is not None:
        # Barrier matrices are conceptually undirected; enforce symmetry to avoid
        # indexing/orientation issues if the file stores only one triangle.
        barrier = barrier_mat.tocsr()
        barrier = 0.5 * (barrier + barrier.T)
        barrier.setdiag(0)
        barrier.eliminate_zeros()
        if barrier.nnz:
            barrier.data = np.clip(barrier.data, 0.0, None)

        bdist = shortest_path(barrier, directed=False, indices=A_idx)
        bab = bdist[:, B_idx_dist]
        finite_bab = bab[np.isfinite(bab)]
        feats["barrier_dist_AB_min"] = float(np.min(finite_bab)) if finite_bab.size else np.nan
        feats["barrier_dist_AB_mean"] = float(np.mean(finite_bab)) if finite_bab.size else np.nan
    else:
        feats["barrier_dist_AB_min"] = np.nan
        feats["barrier_dist_AB_mean"] = np.nan

    return feats


# ======================================================================
#  2. Spectral features
# ======================================================================

def compute_spectral_features(
    Q: csr_matrix,
    pi: np.ndarray,
    n_eigs: int = 10,
) -> Dict[str, float]:
    """
    Compute spectral properties of the CTMC generator.

    Uses the symmetric similarity transform S^{1/2} Q^T S^{-1/2}
    (same approach as mfpt_analysis.py).
    """
    feats: Dict[str, float] = {}
    N = Q.shape[0]
    k = min(n_eigs + 1, N - 1)
    if k <= 1:
        return {f"spectral_{name}": np.nan for name in [
            "gap", "gap_ratio", "fiedler", "ramanujan_score",
            "entropy", "effective_dimension"]}

    pi_safe = np.clip(np.asarray(pi, dtype=float), 1e-300, None)
    sqrt_pi = np.sqrt(pi_safe)
    inv_sqrt_pi = 1.0 / sqrt_pi
    S = diags(sqrt_pi)
    Sinv = diags(inv_sqrt_pi)
    L_sym = S @ Q.T @ Sinv

    # Symmetrize if close
    try:
        from scipy.sparse.linalg import norm as spnorm
        asym = spnorm(L_sym - L_sym.T, ord=1) / max(spnorm(L_sym, ord=1), 1e-300)
        if asym < 1e-8:
            L_sym = 0.5 * (L_sym + L_sym.T)
    except Exception:
        pass

    vals = None
    try:
        vals_raw, _ = eigsh(L_sym, k=k, which="LA", tol=1e-10, maxiter=200000)
        vals = np.real(np.sort(vals_raw)[::-1])
    except (ArpackNoConvergence, Exception):
        for sigma in (-1e-12, -1e-10, -1e-8):
            try:
                vals_raw, _ = eigsh(L_sym, k=k, sigma=sigma, which="LM",
                                    tol=1e-10, maxiter=500000)
                vals = np.real(np.sort(vals_raw)[::-1])
                break
            except Exception:
                continue

    if vals is None:
        return {f"spectral_{name}": np.nan for name in [
            "gap", "gap_ratio", "fiedler", "ramanujan_score",
            "entropy", "effective_dimension"]}

    nonzero = vals[vals < -1e-12]
    if nonzero.size == 0:
        return {f"spectral_{name}": np.nan for name in [
            "gap", "gap_ratio", "fiedler", "ramanujan_score",
            "entropy", "effective_dimension"]}

    lambda1 = nonzero[0]  # closest to 0 (least negative)
    feats["spectral_gap"] = float(-lambda1)  # positive

    if nonzero.size >= 2:
        lambda2 = nonzero[1]
        feats["spectral_gap_ratio"] = float(lambda1 / lambda2)
    else:
        feats["spectral_gap_ratio"] = np.nan

    feats["spectral_fiedler"] = float(-lambda1)  # for CTMC, Fiedler = spectral gap

    # Ramanujan score: spectral_gap / (2*sqrt(d-1)) where d = mean degree
    K_binary = (Q.copy() != 0).astype(float)
    K_binary.setdiag(0)
    K_binary.eliminate_zeros()
    mean_deg = float(K_binary.sum()) / N
    d_eff = max(mean_deg, 2.0)
    feats["spectral_ramanujan_score"] = feats["spectral_gap"] / (2.0 * np.sqrt(d_eff - 1))

    # Spectral entropy over magnitude of eigenvalues
    magnitudes = np.abs(nonzero[:min(n_eigs, nonzero.size)])
    p = magnitudes / magnitudes.sum()
    entropy = -np.sum(p * np.log(p + 1e-300))
    feats["spectral_entropy"] = float(entropy)
    feats["spectral_effective_dimension"] = float(np.exp(entropy))

    return feats


# ======================================================================
#  3. Centrality features
# ======================================================================

def _sparse_pagerank(K: csr_matrix, alpha: float = 0.85, tol: float = 1e-8,
                     max_iter: int = 200) -> np.ndarray:
    """Power iteration PageRank on the rate matrix (column-stochastic)."""
    N = K.shape[0]
    # Normalize columns to get transition matrix
    col_sums = np.asarray(K.sum(axis=0)).ravel()
    col_sums[col_sums == 0] = 1.0
    T_mat = K.multiply(1.0 / col_sums)  # column-stochastic

    pr = np.ones(N) / N
    for _ in range(max_iter):
        pr_new = alpha * (T_mat @ pr) + (1 - alpha) / N
        pr_new /= pr_new.sum()
        if np.linalg.norm(pr_new - pr, 1) < tol:
            break
        pr = pr_new
    return pr_new


def compute_centrality_features(
    K: csr_matrix,
    pi: np.ndarray,
    A_sel: np.ndarray,
    B_sel: np.ndarray,
) -> Dict[str, float]:
    """
    Compute centrality of A and B states in the network.

    Includes PageRank, stationary probability, eigenvector centrality,
    and a closeness-like score on the undirected connectivity graph.
    """
    feats: Dict[str, float] = {}
    N = K.shape[0]

    # Stationary probability of A/B
    feats["pi_A"] = float(pi[A_sel].sum())
    feats["pi_B"] = float(pi[B_sel].sum())
    feats["pi_ratio_AB"] = feats["pi_A"] / max(feats["pi_B"], 1e-300)

    # PageRank
    pr = _sparse_pagerank(K)
    feats["pagerank_A"] = float(pr[A_sel].sum())
    feats["pagerank_B"] = float(pr[B_sel].sum())

    # Eigenvector centrality from dominant eigenvector of adjacency
    try:
        # Use the symmetrized adjacency for eigenvector centrality
        adj = K.copy()
        adj.data[:] = 1.0  # binary adjacency
        adj_sym = 0.5 * (adj + adj.T)
        vals, vecs = eigsh(adj_sym.astype(float), k=1, which="LM")
        ev = np.abs(vecs[:, 0])
        ev /= ev.sum()
        feats["eigvec_centrality_A"] = float(ev[A_sel].sum())
        feats["eigvec_centrality_B"] = float(ev[B_sel].sum())
    except Exception:
        feats["eigvec_centrality_A"] = np.nan
        feats["eigvec_centrality_B"] = np.nan

    # Closeness-like centrality for A/B nodes.
    # We only need scores for A/B, so compute shortest paths from those nodes.
    A_idx = np.where(A_sel)[0]
    B_idx_cent = np.where(B_sel)[0]
    query_idx = np.union1d(A_idx, B_idx_cent)

    # Use unweighted shortest paths on the symmetrized connectivity graph.
    adj_binary = (K != 0).astype(float)
    adj_sym_binary = ((adj_binary + adj_binary.T) > 0).astype(float)

    closeness = np.zeros(N, dtype=float)
    if query_idx.size > 0:
        dist_mat = shortest_path(
            adj_sym_binary,
            directed=False,
            indices=query_idx,
            return_predecessors=False,
        )
        for row_i, src in enumerate(query_idx):
            d = dist_mat[row_i]
            finite = d[np.isfinite(d) & (d > 0)]
            if finite.size > 0:
                closeness[src] = finite.size / finite.sum()

    feats["closeness_centrality_A"] = float(closeness[A_sel].sum())
    feats["closeness_centrality_B"] = float(closeness[B_sel].sum())

    return feats


# ======================================================================
#  4. Community / metastability features
# ======================================================================

def compute_community_features(
    K: csr_matrix,
    pi: np.ndarray,
    A_sel: np.ndarray,
    B_sel: np.ndarray,
    max_clusters: int = 10,
) -> Dict[str, float]:
    """
    Detect metastable communities using spectral clustering on the
    symmetrized rate matrix.
    """
    feats: Dict[str, float] = {}
    N = K.shape[0]

    # Build symmetrized affinity matrix from rates
    K_sym = 0.5 * (K + K.T)
    K_sym.data = np.abs(K_sym.data)  # ensure non-negative
    deg = np.asarray(K_sym.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    # Determine number of clusters from eigengap heuristic
    n_try = min(max_clusters + 1, N - 1, 15)
    if n_try < 2:
        feats["n_communities"] = 1
        feats["modularity"] = 0.0
        feats["AB_same_community"] = 1
        feats["community_size_entropy"] = 0.0
        feats["largest_community_frac"] = 1.0
        return feats

    try:
        # Graph Laplacian eigenvalues for eigengap
        D_inv_sqrt = diags(1.0 / np.sqrt(deg))
        L_norm = diags(np.ones(N)) - D_inv_sqrt @ K_sym @ D_inv_sqrt

        eig_vals, eig_vecs = eigsh(L_norm, k=n_try, which="SM", tol=1e-8)
        eig_vals = np.sort(np.real(eig_vals))

        # Eigengap: largest jump after first eigenvalue (~0)
        gaps = np.diff(eig_vals[1:])  # skip the first ~0 eigenvalue
        if gaps.size > 0:
            n_clusters = int(np.argmax(gaps) + 2)  # +2 because we skipped first and argmax is 0-indexed
            n_clusters = max(2, min(n_clusters, max_clusters))
        else:
            n_clusters = 2
    except Exception:
        n_clusters = 2
        eig_vecs = None

    # Do spectral clustering
    try:
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=42,
        )
        # SpectralClustering needs dense affinity for precomputed
        if N <= 5000:
            labels = sc.fit_predict(K_sym.toarray())
        else:
            # For large networks, use our own spectral embedding + KMeans
            from sklearn.cluster import KMeans
            if eig_vecs is not None and eig_vecs.shape[1] >= n_clusters:
                embedding = eig_vecs[:, :n_clusters]
            else:
                try:
                    _, embedding = eigsh(L_norm, k=n_clusters, which="SM")
                except Exception:
                    embedding = None

            if embedding is not None:
                from sklearn.preprocessing import normalize
                embedding = normalize(embedding, axis=1)
                labels = KMeans(n_clusters=n_clusters, random_state=42,
                                n_init=10).fit_predict(embedding)
            else:
                labels = np.zeros(N, dtype=int)
    except Exception:
        labels = np.zeros(N, dtype=int)

    n_actual = len(set(labels))
    feats["n_communities"] = n_actual

    # Modularity
    m = K_sym.sum() / 2.0
    if m > 0:
        Q_mod = 0.0
        for c in range(n_actual):
            idx_c = np.where(labels == c)[0]
            if idx_c.size == 0:
                continue
            internal = K_sym[idx_c][:, idx_c].sum() / 2.0
            degree_c = deg[idx_c].sum()
            Q_mod += internal / m - (degree_c / (2.0 * m)) ** 2
        feats["modularity"] = float(Q_mod)
    else:
        feats["modularity"] = 0.0

    # A and B in same community?
    A_communities = set(labels[A_sel])
    B_communities = set(labels[B_sel])
    feats["AB_same_community"] = int(len(A_communities & B_communities) > 0)

    # Community size entropy
    sizes = np.bincount(labels).astype(float)
    sizes = sizes[sizes > 0]
    p_sizes = sizes / sizes.sum()
    feats["community_size_entropy"] = float(-np.sum(p_sizes * np.log(p_sizes + 1e-300)))

    feats["largest_community_frac"] = float(sizes.max() / N)

    return feats


# ======================================================================
#  5. Path features
# ======================================================================

def compute_path_features(
    K: csr_matrix,
    A_sel: np.ndarray,
    B_sel: np.ndarray,
) -> Dict[str, float]:
    """
    Analyze shortest paths between A and B on the rate graph.
    """
    feats: Dict[str, float] = {}
    A_idx = np.where(A_sel)[0]
    B_idx = np.where(B_sel)[0]

    nan_feats = {
        "shortest_path_hops_AB": np.nan,
        "rate_shortest_path_AB": np.nan,
        "n_short_paths_AB": np.nan,
        "path_redundancy": np.nan,
    }

    if A_idx.size == 0 or B_idx.size == 0:
        feats.update(nan_feats)
        return feats

    # Hop-count shortest paths (unweighted)
    adj = ((K != 0) + (K.T != 0)).astype(float)
    adj.data[:] = 1.0
    dist = shortest_path(adj, directed=False, indices=A_idx)
    hop_dists = dist[:, B_idx]
    finite_hops = hop_dists[np.isfinite(hop_dists)]

    if finite_hops.size == 0:
        feats.update(nan_feats)
        return feats

    min_hops = float(np.min(finite_hops))
    feats["shortest_path_hops_AB"] = min_hops

    # Count node-disjoint short paths (within 2x shortest) as a proxy for redundancy
    # Use the single representative A and B nodes (first of each)
    a_rep = A_idx[0]
    b_rep = B_idx[0]

    # Rate-weighted shortest path length
    L = _rate_length_matrix(K)
    try:
        d_single = shortest_path(L, directed=True, indices=np.array([a_rep]))
        feats["rate_shortest_path_AB"] = float(d_single[0, b_rep])
    except Exception:
        feats["rate_shortest_path_AB"] = np.nan

    # Estimate path redundancy: number of neighbors of A that also connect to B
    # within a short hop count
    threshold = min_hops * 2
    n_short = int(np.sum(finite_hops <= threshold))
    feats["n_short_paths_AB"] = n_short

    # Path redundancy ratio
    feats["path_redundancy"] = n_short / max(A_idx.size * B_idx.size, 1)

    return feats


# ======================================================================
#  6. Topology features
# ======================================================================

def compute_topology_features(K: csr_matrix) -> Dict[str, float]:
    """Compute basic topological descriptors of the KTN."""
    feats: Dict[str, float] = {}
    N = K.shape[0]
    feats["n_nodes"] = N

    # Directed edges (nonzeros in K, excluding diagonal)
    K_coo = K.tocoo()
    off_diag_mask = K_coo.row != K_coo.col
    n_directed_edges = int(off_diag_mask.sum())
    feats["n_edges_directed"] = n_directed_edges

    # Undirected edges
    adj_sym = ((K != 0) + (K.T != 0)).astype(float)
    adj_sym.setdiag(0)
    adj_sym.eliminate_zeros()
    n_undirected = adj_sym.nnz // 2
    feats["n_edges_undirected"] = n_undirected

    feats["density"] = n_undirected / max(N * (N - 1) / 2, 1)

    # Degree statistics (undirected)
    degrees = np.asarray(adj_sym.sum(axis=1)).ravel()
    feats["degree_mean"] = float(np.mean(degrees))
    feats["degree_std"] = float(np.std(degrees))
    feats["degree_max"] = float(np.max(degrees))
    feats["degree_min"] = float(np.min(degrees))
    if degrees.std() > 0:
        feats["degree_skew"] = float(scipy_skew(degrees))
    else:
        feats["degree_skew"] = 0.0

    # Degree assortativity (Pearson correlation of degrees at edge endpoints)
    adj_coo = adj_sym.tocoo()
    src_deg = degrees[adj_coo.row]
    dst_deg = degrees[adj_coo.col]
    if src_deg.std() > 0 and dst_deg.std() > 0:
        feats["degree_assortativity"] = float(np.corrcoef(src_deg, dst_deg)[0, 1])
    else:
        feats["degree_assortativity"] = 0.0

    # Connected components
    n_comp, comp_labels = connected_components(adj_sym, directed=False)
    feats["n_components"] = n_comp
    if N > 0:
        feats["largest_component_frac"] = float(np.bincount(comp_labels).max() / N)
    else:
        feats["largest_component_frac"] = 0.0

    # Local clustering coefficient (sparse triangle counting)
    # C_i = 2 * triangles(i) / (deg(i) * (deg(i) - 1))
    adj_bin = adj_sym.copy()
    adj_bin.data[:] = 1.0
    A2 = adj_bin @ adj_bin
    # triangles(i) = (A^2 .* A)[i,i] / 2  (element-wise multiply, then diagonal)
    A2_A = A2.multiply(adj_bin)
    triangles = np.asarray(A2_A.sum(axis=1)).ravel() / 2.0
    denom = degrees * (degrees - 1)
    denom[denom == 0] = 1.0
    clustering = 2.0 * triangles / denom
    feats["clustering_coeff_mean"] = float(np.mean(clustering))
    feats["clustering_coeff_std"] = float(np.std(clustering))

    return feats


# ======================================================================
#  Main: extract features for all networks
# ======================================================================

def extract_features_one(
    dps_dir: Path,
    T: float = 300.0,
) -> Dict[str, Any]:
    """Extract all graph features for one coarse-grained KTN."""
    tag = temp_tag(T)
    row: Dict[str, Any] = {
        "dps_dir": str(dps_dir),
        "sequence": dps_dir.parent.name.replace("_nocap", ""),
        "system": dps_dir.name.replace("_nocap", ""),
        "variant": dps_dir.name.split("_")[1] if "_" in dps_dir.name else "",
    }

    # Check if coarse model exists
    markov_dir = dps_dir / f"markov_{tag}"
    coarse_dir = markov_dir / f"GT_kept_{tag}"
    if not coarse_dir.exists():
        row["status"] = "MISSING_COARSE"
        return row

    # Load coarse model
    try:
        B, K, Q, tau, pi = load_markov(dps_dir, T, coarse=True)
    except Exception as e:
        row["status"] = f"LOAD_ERROR: {e}"
        return row

    # Load A/B selectors
    A_sel, B_sel = load_AB_selectors(dps_dir, T, coarse=True)
    if A_sel is None or B_sel is None:
        row["status"] = "MISSING_AB"
        return row

    if A_sel.sum() == 0 or B_sel.sum() == 0:
        row["status"] = "EMPTY_AB"
        return row

    # Try loading barrier matrix (may not exist for coarse model)
    barrier_mat = None
    mp = MarkovFilePaths(dps_dir, T)
    try:
        if mp.barrier_matrix_path.exists():
            from scipy.sparse import load_npz
            barrier_candidate = load_npz(mp.barrier_matrix_path)
            # Barrier matrix is often microscopic; use only when aligned to coarse indexing.
            if barrier_candidate.shape == B.shape:
                barrier_mat = barrier_candidate
    except Exception:
        pass

    # Compute all feature groups — guard each independently so a single
    # failure doesn't lose all features for this network.
    print(f"  [graph_features] {dps_dir.name}: N={Q.shape[0]}, "
          f"|A|={A_sel.sum()}, |B|={B_sel.sum()}")

    warnings_list = []
    for name, fn in [
        ("distance", lambda: compute_distance_features(B, A_sel, B_sel, barrier_mat)),
        ("spectral", lambda: compute_spectral_features(Q, pi)),
        ("centrality", lambda: compute_centrality_features(K, pi, A_sel, B_sel)),
        ("community", lambda: compute_community_features(K, pi, A_sel, B_sel)),
        ("path", lambda: compute_path_features(K, A_sel, B_sel)),
        ("topology", lambda: compute_topology_features(K)),
    ]:
        try:
            row.update(fn())
        except Exception as e:
            warnings_list.append(f"{name}: {type(e).__name__}: {e}")
            print(f"    WARNING [{name}] {type(e).__name__}: {e}")

    row["status"] = "OK" if not warnings_list else f"PARTIAL({'; '.join(warnings_list)})"
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Extract graph-theoretic features for all coarse-grained KTNs."
    )
    parser.add_argument(
        "--T", type=float, default=300.0,
        help="Temperature in K (default: 300).",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("graph_features_coarse_T300K.csv"),
        help="Output CSV path.",
    )
    args = parser.parse_args()

    dps_dirs = iter_dps_dirs()
    print(f"[graph_features] Found {len(dps_dirs)} DPS directories.")

    rows = []
    for i, dps_dir in enumerate(dps_dirs, 1):
        print(f"[graph_features] ({i}/{len(dps_dirs)}) Processing {dps_dir.name}...")
        try:
            row = extract_features_one(dps_dir, args.T)
        except Exception as e:
            row = {"dps_dir": str(dps_dir), "status": f"ERROR: {e}"}
        if row.get("status") != "OK":
            print(f"    -> {row.get('status', 'UNKNOWN')}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[graph_features] Saved {len(df)} rows to {args.out}")

    ok = df[df["status"] == "OK"]
    partial = df[df["status"].str.startswith("PARTIAL", na=False)]
    failed = len(df) - len(ok) - len(partial)
    print(f"[graph_features] {len(ok)} OK, {len(partial)} partial, {failed} skipped/errored.")


if __name__ == "__main__":
    main()
