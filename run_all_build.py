#!/usr/bin/env python

from pathlib import Path
import subprocess

BASE = Path("/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped")
TEMPERATURES = [300.0]   # extend if you want multiple T values

for seq_dir in sorted(BASE.glob("*_nocap")):
    for dps_dir in sorted(seq_dir.glob("*_nocap")):
        if not (dps_dir / "min.data").exists():
            continue
        for T in TEMPERATURES:
            print(f"=== {dps_dir}, T={T} K ===")
            cmd = [
                "python",
                "build_markov_model.py",
                "--data-dir",
                str(dps_dir),
                "--T",
                str(T),
            ]
            subprocess.run(cmd, check=True)
