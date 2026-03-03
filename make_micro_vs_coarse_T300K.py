#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz


def get_key(d, keys):
    for k in keys:
        if k in d:
            return float(np.asarray(d[k]))
    return np.nan


def load_timescales(ts_path: Path, k: int = 5):
    if not ts_path.exists():
        return [np.nan] * k
    t = np.load(ts_path).astype(float).ravel()
    out = list(t[:k]) + [np.nan] * max(0, k - len(t))
    return out[:k]


def main():
    root = Path("/scratch/gpfs/JERELLE/harry/thesis_data/LAMMPS_uncapped")
    T = 300
    tag = f"T{T}K"

    rows = []
    for markov_dir in sorted(root.glob(f"*/*/markov_{tag}")):
        dps_dir = markov_dir.parent

        system = markov_dir.parts[-3]   # e.g. aaaaaa_nocap
        dps_name = dps_dir.name         # e.g. aaaaaa_99idps_nocap

        micro_npz = markov_dir / f"AB_kinetics_{tag}.npz"
        eff_dir   = markov_dir / f"GT_kept_{tag}"
        eff_npz   = eff_dir / f"AB_kinetics_{tag}.npz"

        Q_micro_path = markov_dir / f"Q_{tag}.npz"
        Q_eff_path   = eff_dir / f"Q_eff_{tag}.npz"

        # sizes
        N_micro = load_npz(Q_micro_path).shape[0] if Q_micro_path.exists() else np.nan
        N_eff   = load_npz(Q_eff_path).shape[0] if Q_eff_path.exists() else np.nan
        compression = (N_micro / N_eff) if np.isfinite(N_micro) and np.isfinite(N_eff) and N_eff else np.nan

        # coarse timescales (if computed)
        t1, t2, t3, t4, t5 = load_timescales(eff_dir / f"timescales_{tag}.npy", k=5)

        if not micro_npz.exists() or not eff_npz.exists():
            rows.append(dict(
                system=system, dps=dps_name, tag=tag,
                N_micro=N_micro, N_eff=N_eff, compression=compression,
                status="MISSING_NPZ",
                t1=t1, t2=t2, t3=t3, t4=t4, t5=t5,
            ))
            continue

        micro = np.load(micro_npz, allow_pickle=True)
        eff   = np.load(eff_npz, allow_pickle=True)

        mfpt_micro_AB = get_key(micro, ["MFPT_AB", "MFPTAB"])
        mfpt_micro_BA = get_key(micro, ["MFPT_BA", "MFPTBA"])
        mfpt_eff_AB   = get_key(eff,   ["MFPT_AB", "MFPTAB"])
        mfpt_eff_BA   = get_key(eff,   ["MFPT_BA", "MFPTBA"])

        def relerr(a, b):
            if not np.isfinite(a) or not np.isfinite(b) or a == 0:
                return np.nan
            return abs(b - a) / abs(a)

        rows.append(dict(
            system=system, dps=dps_name, tag=tag,
            N_micro=N_micro, N_eff=N_eff, compression=compression,
            MFPT_micro_AB=mfpt_micro_AB, MFPT_eff_AB=mfpt_eff_AB, relerr_AB=relerr(mfpt_micro_AB, mfpt_eff_AB),
            MFPT_micro_BA=mfpt_micro_BA, MFPT_eff_BA=mfpt_eff_BA, relerr_BA=relerr(mfpt_micro_BA, mfpt_eff_BA),
            status="OK",
            t1=t1, t2=t2, t3=t3, t4=t4, t5=t5,
        ))

    df = pd.DataFrame(rows)

    out_csv = Path(f"micro_vs_coarse_{tag}.csv")
    df.to_csv(out_csv, index=False)

    ok = df[df["status"] == "OK"].copy()

    out_txt = Path(f"micro_vs_coarse_{tag}_summary.txt")
    with open(out_txt, "w") as f:
        f.write(f"Micro vs Coarse report ({tag})\n")
        f.write(f"Total rows: {len(df)}\n")
        f.write(str(df['status'].value_counts()) + "\n\n")

        if len(ok) > 0:
            f.write("Compression (N_micro/N_eff):\n")
            f.write(ok["compression"].describe().to_string() + "\n\n")
            f.write("relerr_AB:\n")
            f.write(ok["relerr_AB"].describe().to_string() + "\n\n")
            f.write("relerr_BA:\n")
            f.write(ok["relerr_BA"].describe().to_string() + "\n\n")

            f.write("Coarse timescale t1 (if available):\n")
            f.write(ok["t1"].describe().to_string() + "\n\n")

            f.write("Worst 10 by relerr_AB:\n")
            f.write(ok.sort_values("relerr_AB", ascending=False).head(10)[
                ["system","dps","N_micro","N_eff","compression","relerr_AB","relerr_BA","t1","t2","t3"]
            ].to_string(index=False) + "\n")

    print("Wrote:", out_csv.resolve())
    print("Wrote:", out_txt.resolve())


if __name__ == "__main__":
    main()
