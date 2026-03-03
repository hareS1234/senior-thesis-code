# run_all_for_one_sequence.py

"""
Example script: for ONE sequence (e.g. YYGGYY monomer), do:

1. Load minima + TS files.
2. Build graphs + matrices.
3. Compute Dijkstra distances, Ramanujan score, Markov stats.
4. Output per-sequence summary CSV.

You can clone this script for each sequence, or loop over many sequences.
"""

from __future__ import annotations
import os
import json

import pandas as pd

from config import PHYS
from io_disconnection import load_minima, load_transition_states
from build_networks import build_graphs_and_matrices
from graph_metrics import compute_all_pairs_dijkstra, compute_basic_graph_stats
from summaries_and_regression import build_sequence_summary, add_graph_stats_to_summary


def run_for_sequence(
    sequence_id: str,
    minima_path: str,
    ts_path: str,
    out_dir: str = "results",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load data
    minima = load_minima(minima_path)
    ts = load_transition_states(ts_path)

    # 2) Build networks and matrices
    G_und, G_rate, A, Q = build_graphs_and_matrices(
        minima=minima,
        ts=ts,
        temperature=PHYS.temperature,
        k0=PHYS.k0,
    )

    # 3) Distances + graph metrics
    dists = compute_all_pairs_dijkstra(G_und)
    g_stats = compute_basic_graph_stats(G_und)

    # 4) Summary features
    summary = build_sequence_summary(
        sequence_id=sequence_id,
        minima=minima,
        A=A,
        Q=Q,
        dists=dists,
    )
    summary = add_graph_stats_to_summary(summary, g_stats)

    # 5) Save everything
    # Summary as CSV row
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(out_dir, f"{sequence_id}_summary.csv"), index=False)

    # Optionally save matrices / distances as numpy
    import numpy as np

    np.save(os.path.join(out_dir, f"{sequence_id}_A.npy"), A)
    np.save(os.path.join(out_dir, f"{sequence_id}_Q.npy"), Q)
    np.save(os.path.join(out_dir, f"{sequence_id}_dists.npy"), dists)

    # metadata about how it was computed
    meta = {
        "sequence_id": sequence_id,
        "minima_path": minima_path,
        "ts_path": ts_path,
        "temperature": PHYS.temperature,
        "k0": PHYS.k0,
        "n_minima": int(len(minima)),
        "n_ts": int(len(ts)),
    }
    with open(os.path.join(out_dir, f"{sequence_id}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Finished sequence {sequence_id}. Summary + matrices written to {out_dir}/")


if __name__ == "__main__":
    # EXAMPLE: tweak file names to your actual ones
    run_for_sequence(
        sequence_id="YYGGYY_monomer",
        minima_path="data/YYGGYY_monomer_minima.dat",
        ts_path="data/YYGGYY_monomer_ts.dat",
        out_dir="results_YYGGYY_monomer",
    )
