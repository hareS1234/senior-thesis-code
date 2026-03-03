# graph_metrics.py

from __future__ import annotations
from typing import Dict, Any

import numpy as np
import networkx as nx

from config import PIPE


def compute_all_pairs_dijkstra(
    G: nx.Graph,
    weight: str | None = None
) -> np.ndarray:
    """
    Compute all-pairs shortest path distances using Dijkstra.
    Returns an (n, n) numpy array.
    """
    if weight is None:
        weight = PIPE.dijkstra_weight_key

    # Use Floyd-Warshall for simplicity; on big graphs you can swap to all_pairs_dijkstra
    dist_mat = nx.floyd_warshall_numpy(G, weight=weight)
    return np.array(dist_mat, dtype=float)


def compute_ramanujan_score(A: np.ndarray) -> float:
    """
    Ramanujan-like score for possibly irregular graphs.

    For a d-regular Ramanujan graph, nontrivial eigenvalues λ satisfy:
      |λ| <= 2 * sqrt(d - 1)

    Here we:
      - Compute eigenvalues of A
      - Let λ1 >= |λ2| >= ... be sorted by magnitude
      - Take average degree d_bar
      - Define score = max_{i>=2} |λ_i| / (2 * sqrt(d_bar - 1))

    Lower score ~ more 'Ramanujan-like' (strong expander).
    """
    if A.shape[0] == 0:
        return np.nan

    # Degrees from adjacency
    deg = A.sum(axis=1)
    d_bar = float(deg.mean())
    if d_bar <= 1:
        return np.nan

    vals = np.linalg.eigvals(A)
    vals = np.sort(np.abs(vals))[::-1]  # descending
    if len(vals) < 2:
        return np.nan
    lam_nontriv_max = float(vals[1])  # largest nontrivial eigenvalue magnitude

    bound = 2.0 * np.sqrt(d_bar - 1.0)
    score = lam_nontriv_max / bound
    return score


def compute_basic_graph_stats(G: nx.Graph) -> Dict[str, Any]:
    """
    Some basic network metrics useful as sequence-level features.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return {
            "n_nodes": 0,
            "n_edges": 0,
            "avg_degree": np.nan,
            "avg_clustering": np.nan,
            "avg_shortest_path": np.nan,
        }

    degrees = [deg for _, deg in G.degree()]
    avg_degree = float(np.mean(degrees))

    try:
        avg_clustering = nx.average_clustering(G, weight=None)
    except Exception:
        avg_clustering = np.nan

    # Use largest connected component for path length
    if nx.is_connected(G):
        try:
            avg_shortest_path = nx.average_shortest_path_length(G, weight=PIPE.dijkstra_weight_key)
        except Exception:
            avg_shortest_path = np.nan
    else:
        # restrict to LCC
        cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(cc)
        try:
            avg_shortest_path = nx.average_shortest_path_length(H, weight=PIPE.dijkstra_weight_key)
        except Exception:
            avg_shortest_path = np.nan

    return {
        "n_nodes": n,
        "n_edges": m,
        "avg_degree": avg_degree,
        "avg_clustering": float(avg_clustering),
        "avg_shortest_path": float(avg_shortest_path),
    }
