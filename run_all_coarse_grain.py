#!/usr/bin/env python
"""
run_all_coarse_grain.py

For every DPS directory under BASE_DIR:

1. Ensure keep_minima.txt exists (auto-generate from min.A/min.B + low-E minima).
2. For each temperature T, run coarse_grain_markov.py to build the GT/NGT
   coarse-grained Markov model in markov_TxxxK/GT_kept_TxxxK/.

This assumes:
- build_markov_model.py has already been run for each DPS/T.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from config import BASE_DIR, SEQUENCE_GLOB, DPS_GLOB, TEMPERATURES
from generate_keep_list import build_keep_set, write_keep_file
from io_markov import temp_tag, markov_dir_for_T


def iter_dps_dirs() -> List[Path]:
    """Find all DPS directories under BASE_DIR that contain min.data."""
    dps_dirs: List[Path] = []
    for seq_dir in BASE_DIR.glob(SEQUENCE_GLOB):
        if not seq_dir.is_dir():
            continue
        for dps_dir in seq_dir.glob(DPS_GLOB):
            if not dps_dir.is_dir():
                continue
            if (dps_dir / "min.data").exists() and (dps_dir / "ts.data").exists():
                dps_dirs.append(dps_dir)
    return sorted(dps_dirs)


def run_cmd(cmd: List[str]):
    print(f"[run_all_coarse_grain] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with code {res.returncode}: {' '.join(cmd)}")


def main():
    dps_dirs = iter_dps_dirs()
    print(f"[run_all_coarse_grain] Found {len(dps_dirs)} DPS directories under {BASE_DIR}")

    for dps_dir in dps_dirs:
        print(f"\n=== DPS directory: {dps_dir} ===")

        # 1. Ensure keep_minima.txt exists
        keep_path = dps_dir / "keep_minima.txt"
        if not keep_path.exists():
            keep_ids = build_keep_set(dps_dir, extra_lowE=10, E_window=None)
            write_keep_file(dps_dir, keep_ids, overwrite=False)
        else:
            print(f"[run_all_coarse_grain] Using existing keep_minima.txt in {dps_dir}")

        # 2. Coarse-grain for each temperature
        for T in TEMPERATURES:
            tag = temp_tag(T)
            markov_dir = markov_dir_for_T(dps_dir, T)
            if not markov_dir.exists():
                print(f"[run_all_coarse_grain] WARNING: {markov_dir} not found; "
                      f"run build_markov_model.py first.")
                continue

            cmd = [
                "python",
                "coarse_grain_markov.py",
                "--markov-dir",
                str(markov_dir),
                "--T",
                str(T),
                "--keep-file",
                str(keep_path),
            ]
            run_cmd(cmd)

    print("\n[run_all_coarse_grain] All done.")


if __name__ == "__main__":
    main()
