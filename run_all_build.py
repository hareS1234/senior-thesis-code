#!/usr/bin/env python
"""Build microscopic Markov models for every DPS directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from config import BASE_DIR, TEMPERATURES, MarkovFilePaths, iter_dps_dirs


def model_is_complete(dps_dir: Path, temperature: float) -> bool:
    paths = MarkovFilePaths(dps_dir, temperature)
    needed = (
        paths.B_path,
        paths.K_path,
        paths.Q_path,
        paths.tau_path,
        paths.pi_path,
        paths.energies_path,
        paths.entropies_path,
        paths.retained_mask_path,
        paths.orig_ids_path,
    )
    return all(path.exists() for path in needed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build microscopic Markov models for every DPS directory."
    )
    parser.add_argument("--root", type=Path, default=BASE_DIR)
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dps_dirs = iter_dps_dirs(args.root.resolve())
    print(f"[run_all_build] Found {len(dps_dirs)} DPS directories.")

    built = 0
    skipped = 0
    build_script = Path(__file__).with_name("build_markov_model.py")
    for dps_dir in dps_dirs:
        for temperature in args.temperatures:
            if model_is_complete(dps_dir, temperature) and not args.overwrite:
                print(f"[run_all_build] Skipping complete model: {dps_dir}")
                skipped += 1
                continue

            print(f"=== {dps_dir}, T={temperature} K ===")
            cmd = [
                sys.executable,
                str(build_script),
                "--data-dir",
                str(dps_dir),
                "--T",
                str(temperature),
            ]
            if args.quiet:
                cmd.append("--quiet")
            subprocess.run(cmd, check=True)
            built += 1

    print(f"[run_all_build] Done. Built {built}; skipped {skipped}.")


if __name__ == "__main__":
    main()
