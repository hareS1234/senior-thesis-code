#!/usr/bin/env python3
"""
make_micro_report.py

Scan a root folder (e.g. .../LAMMPS_uncapped) for fine-grained markov_T*K folders
and generate a clear report of kinetic + validation numbers.

Outputs:
  - micro_report.csv : machine-readable table
  - micro_report.txt : human-readable report grouped by sequence and T

Designed for your directory structure like:
  LAMMPS_uncapped/<seq>_nocap/<seq>_99idps_nocap/markov_T300K/
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import load_npz


# ----------------------------
# Small utilities
# ----------------------------

def parse_tag_from_markov_dir(markov_dir: Path) -> str:
    # markov_dir.name like "markov_T300K"
    name = markov_dir.name
    if not name.startswith("markov_"):
        raise ValueError(f"Not a markov dir: {markov_dir}")
    return name.split("markov_", 1)[1]  # "T300K"


def parse_T_from_tag(tag: str) -> Optional[int]:
    # tag like "T300K"
    try:
        if tag.startswith("T") and tag.endswith("K"):
            return int(tag[1:-1])
    except Exception:
        return None
    return None


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def read_min_set(path: Path) -> np.ndarray:
    """Read min.A / min.B (PATHSAMPLE style). Handles optional count in first entry."""
    if not path.exists():
        return np.array([], dtype=int)
    data = np.loadtxt(path, dtype=int, ndmin=1)
    data = np.atleast_1d(data).ravel()
    if data.size == 0:
        return data
    first = int(data[0])
    n_rest = data.size - 1
    # PATHSAMPLE convention: first entry equals number of IDs that follow
    if n_rest == first:
        return data[1:]
    return data


def load_orig_ids(markov_dir: Path, tag: str) -> Optional[np.ndarray]:
    cand = [
        markov_dir / f"original_min_ids_{tag}.npy",
        markov_dir / f"orig_min_ids_{tag}.npy",
    ]
    p = find_first_existing(cand)
    if p is None:
        return None
    return np.load(p)


def load_AB_selectors(markov_dir: Path, dps_dir: Path, tag: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prefer A_states/B_states in markov_dir (already aligned to Q indexing).
    Fall back to min.A/min.B + original_min_ids mapping if needed.
    """
    A_states = markov_dir / f"A_states_{tag}.npy"
    B_states = markov_dir / f"B_states_{tag}.npy"
    if A_states.exists() and B_states.exists():
        A = np.load(A_states).astype(bool)
        B = np.load(B_states).astype(bool)
        return A, B

    orig_ids = load_orig_ids(markov_dir, tag)
    if orig_ids is None:
        return None, None

    A_ids = read_min_set(dps_dir / "min.A")
    B_ids = read_min_set(dps_dir / "min.B")
    if A_ids.size == 0 or B_ids.size == 0:
        return None, None

    A_set = set(int(x) for x in A_ids)
    B_set = set(int(x) for x in B_ids)

    A = np.array([int(i) in A_set for i in orig_ids], dtype=bool)
    B = np.array([int(i) in B_set for i in orig_ids], dtype=bool)
    return A, B


def load_mfpt_npz(markov_dir: Path, tag: str) -> Dict[str, float]:
    """
    Load MFPT info from AB_kinetics_{tag}.npz, handling key name variations.
    """
    out: Dict[str, float] = {}
    f = markov_dir / f"AB_kinetics_{tag}.npz"
    if not f.exists():
        return out

    d = np.load(f, allow_pickle=True)

    # common fields
    for k in d.keys():
        v = d[k]
        try:
            out[k] = float(np.asarray(v))
        except Exception:
            pass

    # normalize MFPT keys into MFPT_AB / MFPT_BA
    if "MFPTAB" in out and "MFPT_AB" not in out:
        out["MFPT_AB"] = out["MFPTAB"]
    if "MFPTBA" in out and "MFPT_BA" not in out:
        out["MFPT_BA"] = out["MFPTBA"]

    return out


def stationarity_metrics(Q, pi) -> Tuple[float, float, float]:
    """
    Return (||Q pi||_1, || |Q| pi ||_1, relative).
    """
    qpi = Q @ pi
    res = float(np.linalg.norm(np.asarray(qpi).ravel(), 1))

    Qabs = Q.copy()
    Qabs.data = np.abs(Qabs.data)
    scale_vec = Qabs @ pi
    scale = float(np.linalg.norm(np.asarray(scale_vec).ravel(), 1))

    rel = res / scale if scale > 0 else float("nan")
    return res, scale, rel


def generator_sanity(Q, tol: float = 1e-12) -> Dict[str, float]:
    """
    Check sign pattern + conservation style diagnostics.
    """
    info: Dict[str, float] = {}
    Qcoo = Q.tocoo()
    diag_mask = (Qcoo.row == Qcoo.col)
    diag = Qcoo.data[diag_mask]
    off = Qcoo.data[~diag_mask]

    info["diag_pos_count"] = int(np.sum(diag > tol))
    info["off_neg_count"] = int(np.sum(off < -tol))

    row_sums = np.asarray(Q.sum(axis=1)).ravel()
    col_sums = np.asarray(Q.sum(axis=0)).ravel()
    info["max_abs_rowsum"] = float(np.max(np.abs(row_sums))) if row_sums.size else 0.0
    info["max_abs_colsum"] = float(np.max(np.abs(col_sums))) if col_sums.size else 0.0

    return info


def infer_system_label(root: Path, markov_dir: Path) -> str:
    """
    Use the first directory under root as the system label
    (e.g. aaaaaa_nocap, yyggyy_dimer, etc.).
    """
    rel = markov_dir.relative_to(root)
    if len(rel.parts) == 0:
        return markov_dir.parent.name
    return rel.parts[0]


# ----------------------------
# Main crawl + report
# ----------------------------

def collect_one(markov_dir: Path, root: Path) -> Dict[str, object]:
    tag = parse_tag_from_markov_dir(markov_dir)
    T = parse_T_from_tag(tag)

    dps_dir = markov_dir.parent  # e.g. .../aaaaaa_99idps_nocap
    system = infer_system_label(root, markov_dir)
    sequence = system.replace("_nocap", "")

    Q_path = markov_dir / f"Q_{tag}.npz"
    pi_path = markov_dir / f"pi_{tag}.npy"

    row: Dict[str, object] = {
        "system": system,
        "sequence": sequence,
        "tag": tag,
        "T_K": T if T is not None else "",
        "markov_dir": str(markov_dir),
        "dps_dir": str(dps_dir),
    }

    if not Q_path.exists() or not pi_path.exists():
        row["status"] = "MISSING_Q_OR_PI"
        return row

    Q = load_npz(Q_path).tocsr()
    pi = np.load(pi_path).astype(float)

    N = int(Q.shape[0])
    row["N"] = N
    row["nnz_Q"] = int(Q.nnz)
    row["nnz_per_state"] = float(Q.nnz) / float(N) if N > 0 else float("nan")
    row["pi_sum"] = float(pi.sum())
    row["pi_min"] = float(pi.min()) if pi.size else float("nan")
    row["pi_max"] = float(pi.max()) if pi.size else float("nan")

    # Generator sanity
    san = generator_sanity(Q)
    row.update(san)

    # Stationarity
    res, scale, rel = stationarity_metrics(Q, pi)
    row["Qpi_norm1"] = res
    row["absQpi_norm1"] = scale
    row["Qpi_rel"] = rel

    # A/B selectors
    A_sel, B_sel = load_AB_selectors(markov_dir, dps_dir, tag)
    if A_sel is None or B_sel is None:
        row["nA"] = ""
        row["nB"] = ""
        row["AB_overlap"] = ""
    else:
        nA = int(np.sum(A_sel))
        nB = int(np.sum(B_sel))
        row["nA"] = nA
        row["nB"] = nB
        row["AB_overlap"] = int(np.sum(A_sel & B_sel))

        # singleton indices (useful for debugging + reporting)
        if nA == 1:
            row["A_index"] = int(np.where(A_sel)[0][0])
        else:
            row["A_index"] = ""
        if nB == 1:
            row["B_index"] = int(np.where(B_sel)[0][0])
        else:
            row["B_index"] = ""

    # MFPTs from saved file
    mfpt = load_mfpt_npz(markov_dir, tag)
    row["MFPT_AB"] = mfpt.get("MFPT_AB", "")
    row["MFPT_BA"] = mfpt.get("MFPT_BA", "")
    row["Var_AB"] = mfpt.get("Var_AB", "")
    row["Var_BA"] = mfpt.get("Var_BA", "")

    # Status flag (simple)
    ok = True
    ok &= (row["off_neg_count"] == 0)
    ok &= (row["diag_pos_count"] == 0)
    ok &= (math.isfinite(row["Qpi_rel"]) and row["Qpi_rel"] < 1e-10)
    # For your convention, column sums should be ~0; we don't force a hard tol here,
    # but we record it.
    row["status"] = "OK" if ok else "CHECK"

    return row


def main():
    ap = argparse.ArgumentParser(description="Generate fine-grained micro KTN report from LAMMPS_uncapped.")
    ap.add_argument("--root", type=Path, required=True,
                    help="Root directory to scan (e.g. /scratch/gpfs/.../LAMMPS_uncapped).")
    ap.add_argument("--out-prefix", type=str, default="micro_report",
                    help="Output prefix (writes <prefix>.csv and <prefix>.txt).")
    ap.add_argument("--tag", type=str, default="",
                    help="Optional: only include a specific tag (e.g. T300K).")
    args = ap.parse_args()

    root = args.root.resolve()
    tag_filter = args.tag.strip()

    # Find all markov_T*K directories
    markov_dirs = []
    for p in root.rglob("markov_T*K"):
        if p.is_dir():
            if tag_filter and p.name != f"markov_{tag_filter}":
                continue
            markov_dirs.append(p)

    markov_dirs.sort()

    rows = []
    for md in markov_dirs:
        try:
            rows.append(collect_one(md, root))
        except Exception as e:
            rows.append({
                "markov_dir": str(md),
                "status": f"ERROR: {type(e).__name__}: {e}",
            })

    # Write CSV
    csv_path = Path(f"{args.out_prefix}.csv")
    # union of keys
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Write human-readable TXT
    txt_path = Path(f"{args.out_prefix}.txt")
    with txt_path.open("w") as fh:
        fh.write(f"Micro KTN report for root: {root}\n")
        fh.write(f"Found {len(rows)} markov_T*K folders\n\n")

        # summary counts
        status_counts = {}
        for r in rows:
            status_counts[r.get("status", "UNKNOWN")] = status_counts.get(r.get("status", "UNKNOWN"), 0) + 1
        fh.write("Status counts:\n")
        for k, v in sorted(status_counts.items()):
            fh.write(f"  {k:20s} {v}\n")
        fh.write("\n")

        # group by system then tag
        rows_sorted = sorted(rows, key=lambda r: (str(r.get("system","")), str(r.get("tag",""))))
        current = None
        for r in rows_sorted:
            header = (r.get("system","?"), r.get("tag","?"))
            if header != current:
                current = header
                fh.write("="*80 + "\n")
                fh.write(f"{header[0]}  |  {header[1]}\n")
                fh.write("="*80 + "\n")

            fh.write(f"markov_dir: {r.get('markov_dir','')}\n")
            fh.write(f"status    : {r.get('status','')}\n")
            fh.write(f"N         : {r.get('N','')}   nnz(Q): {r.get('nnz_Q','')}   nnz/state: {r.get('nnz_per_state','')}\n")
            fh.write(f"pi sum    : {r.get('pi_sum','')}   min: {r.get('pi_min','')}   max: {r.get('pi_max','')}\n")
            fh.write(f"max|colsum|: {r.get('max_abs_colsum','')}   max|rowsum|: {r.get('max_abs_rowsum','')}\n")
            fh.write(f"sign checks: off_neg={r.get('off_neg_count','')}  diag_pos={r.get('diag_pos_count','')}\n")
            fh.write(f"stationary : ||Qpi||1={r.get('Qpi_norm1','')}  rel={r.get('Qpi_rel','')}\n")
            fh.write(f"A/B        : nA={r.get('nA','')} (idx {r.get('A_index','')})   nB={r.get('nB','')} (idx {r.get('B_index','')})   overlap={r.get('AB_overlap','')}\n")
            fh.write(f"MFPTs      : A->B={r.get('MFPT_AB','')}   B->A={r.get('MFPT_BA','')}\n")
            if r.get("Var_AB","") != "" or r.get("Var_BA","") != "":
                fh.write(f"Vars       : Var_AB={r.get('Var_AB','')}   Var_BA={r.get('Var_BA','')}\n")
            fh.write("\n")

    print(f"Wrote: {csv_path.resolve()}")
    print(f"Wrote: {txt_path.resolve()}")


if __name__ == "__main__":
    main()
