# build_networks.py

from __future__ import annotations
from typing import Tuple

import numpy as np
import networkx as nx
import pandas as pd

from config import PHYS, PIPE
from io_disconnection import build_min_id_index


def build_graphs_and_matrices(
    minima: pd.DataFrame,
    ts: pd.DataFrame,
    temperature: float | None = None,
    k0: float | None = None,
) -> Tuple[nx.Graph, nx.DiGraph, np.ndarray, np.ndarray]:
    """
    From minima + TS tables, build:
      - undirected barrier graph (G_undirected)
      - directed rate graph (G_rate)
      - adjacency matrix A (undirected, symmetric)
      - Markov generator Q (row-sum = 0)

    Rates use a simple Arrhenius-like form:
      k_ij = k0 * exp( - (E_ts - E_i) / (kb T) )

    You can change this later.
    """
    if temperature is None:
        temperature = PHYS.temperature
    if k0 is None:
        k0 = PHYS.k0

    kb = PHYS.kb

    n_min = len(minima)
    id_to_idx = build_min_id_index(minima)
    energies = minima.set_index("min_id")["energy"].to_dict()

    # Graphs
    G_undirected = nx.Graph()
    G_rate = nx.DiGraph()
    G_undirected.add_nodes_from(range(n_min))
    G_rate.add_nodes_from(range(n_min))

    # Matrices
    A = np.zeros((n_min, n_min), dtype=float)
    Q = np.zeros((n_min, n_min), dtype=float)

    for _, row in ts.iterrows():
        m1_id = int(row["min1"])
        m2_id = int(row["min2"])
        ts_e = float(row["ts_energy"])

        if m1_id not in id_to_idx or m2_id not in id_to_idx:
            # Skip TS that references minima we didn't load
            continue

        i = id_to_idx[m1_id]
        j = id_to_idx[m2_id]

        E_i = energies[m1_id]
        E_j = energies[m2_id]

        # Barrier heights relative to each minimum
        ΔE_i = ts_e - E_i
        ΔE_j = ts_e - E_j

        # Simple Arrhenius rates
        k_ij = k0 * np.exp(-ΔE_i / (kb * temperature))
        k_ji = k0 * np.exp(-ΔE_j / (kb * temperature))

        # Undirected graph weight: barrier height above lower minimum
        barrier_height = ts_e - min(E_i, E_j)
        w = max(barrier_height, 1e-12)  # avoid zero; you can choose other scheme

        # Update adjacency matrix (take smallest barrier if multiple TS)
        if A[i, j] == 0 or w < A[i, j]:
            A[i, j] = A[j, i] = w

        # Update Q (Markov generator)
        Q[i, j] += k_ij
        Q[j, i] += k_ji

        # Graph edges
        G_undirected.add_edge(i, j, weight=w, ts_energy=ts_e)
        G_rate.add_edge(i, j, rate=k_ij)
        G_rate.add_edge(j, i, rate=k_ji)

    # Set diagonal of Q to make row sums zero
    for i in range(n_min):
        Q[i, i] = -np.sum(Q[i, :]) + Q[i, i]  # keep any diagonal we already set

    return G_undirected, G_rate, A, Q
