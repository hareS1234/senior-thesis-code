"""
run_all_sequences.py

Driver script to:
- iterate over all DPS directories under BASE_DIR
- (optionally) call build_markov_model.py and coarse_grain_markov.py
- run markov_analysis.py for each
- aggregate per-run JSON summaries into one CSV for later regression.

Run from the directory containing all the Python scripts.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd  # you should `conda install pandas` in your thesis env

from config import (
    BASE_DIR,
    TEMPERATURES,
    iter_dps_dirs,
    MarkovFilePaths,
)


THIS_DIR = Path(__file__).resolve().parent


def run_cmd(cmd: List[str], cwd: Path | None = None):
    """Run a command and stream output; raise on error."""
    print(f"\n[run_all] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def process_all_sequences(
    temperatures: List[float] | None = None,
    rebuild_markov: bool = False,
    redo_analysis: bool = True,
) -> pd.DataFrame:
    """
    Main entry point.

    Parameters
    ----------
    temperatures : list of float
        Temperatures to process; default uses TEMPERATURES from config.
    rebuild_markov : bool
        If True, call build_markov_model.py and coarse_grain_markov.py for each DPS dir.
    redo_analysis : bool
        If False, skip markov_analysis if the summary JSON already exists.

    Returns
    -------
    df : pandas.DataFrame
        Aggregated summaries for all runs / temperatures.
    """
    if temperatures is None:
        temperatures = TEMPERATURES

    dps_dirs = iter_dps_dirs()
    print(f"[run_all] Found {len(dps_dirs)} DPS directories under {BASE_DIR}")

    rows = []

    for dps_dir in dps_dirs:
        for T in temperatures:
            mp = MarkovFilePaths(dps_dir, T)

            if rebuild_markov:
                run_cmd(
                    [
                        "python",
                        str(THIS_DIR / "build_markov_model.py"),
                        str(dps_dir),
                        "--T",
                        str(T),
                    ]
                )
                run_cmd(
                    [
                        "python",
                        str(THIS_DIR / "coarse_grain_markov.py"),
                        str(dps_dir),
                        "--T",
                        str(T),
                    ]
                )

            if redo_analysis or not mp.summary_json_path.exists():
                run_cmd(
                    [
                        "python",
                        str(THIS_DIR / "markov_analysis.py"),
                        str(dps_dir),
                        "--T",
                        str(T),
                    ]
                )

            # Load summary JSON
            with open(mp.summary_json_path, "r") as fh:
                summary = json.load(fh)
            rows.append(summary)

    df = pd.DataFrame(rows)
    out_csv = THIS_DIR / "all_sequences_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"[run_all] Wrote aggregated summary to {out_csv}")
    return df


if __name__ == "__main__":
    process_all_sequences()
