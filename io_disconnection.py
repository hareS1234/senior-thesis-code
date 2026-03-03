# io_disconnection.py

"""
Parsing utilities for minima and transition-state (TS) data exported from
disconnectivity graphs / DPS codes.

ASSUMED FORMATS (customize to your actual files):

1) Minima file (whitespace-separated, comments start with #):
   min_id   energy   [optional other columns...]

2) Transition-state file (whitespace-separated, comments start with #):
   ts_id   ts_energy   min1_id   min2_id   [optional other columns...]
"""

from __future__ import annotations
from typing import Tuple, Dict

import pandas as pd


def _read_table(path: str, n_cols_min: int) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < n_cols_min:
                continue
            rows.append(parts)
    if not rows:
        raise ValueError(f"No data parsed from {path}")
    return pd.DataFrame(rows)


def load_minima(path: str) -> pd.DataFrame:
    """
    Load minima from a DPS-style file.
    Assumes: col0 = min_id, col1 = energy.
    """
    df_raw = _read_table(path, n_cols_min=2)
    df_raw.columns = [f"col_{i}" for i in range(df_raw.shape[1])]

    minima = pd.DataFrame({
        "min_id": df_raw["col_0"].astype(int),
        "energy": df_raw["col_1"].astype(float),
    })
    return minima.sort_values("min_id").reset_index(drop=True)


def load_transition_states(path: str) -> pd.DataFrame:
    """
    Load TS data from a DPS-style file.
    Assumes: col0 = ts_id, col1 = ts_energy, col2 = min1_id, col3 = min2_id.
    """
    df_raw = _read_table(path, n_cols_min=4)
    df_raw.columns = [f"col_{i}" for i in range(df_raw.shape[1])]

    ts = pd.DataFrame({
        "ts_id": df_raw["col_0"].astype(int),
        "ts_energy": df_raw["col_1"].astype(float),
        "min1": df_raw["col_2"].astype(int),
        "min2": df_raw["col_3"].astype(int),
    })
    return ts.reset_index(drop=True)


def build_min_id_index(minima: pd.DataFrame) -> Dict[int, int]:
    """
    Map minimum ID (from file) -> contiguous index [0, ..., n_min-1].
    """
    return {mid: i for i, mid in enumerate(minima["min_id"].values)}
