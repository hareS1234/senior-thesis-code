#!/usr/bin/env python3
"""Compare the microscopic and GT-kept models at 300 K.

Writes one combined CSV plus a short summary for the later analysis scripts.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from scipy.sparse import load_npz, csr_matrix
from scipy.sparse.csgraph import connected_components






def parse_tag(T: int) -> str:
    return f"T{int(T)}K"


def safe_load_npz(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=True))


def get_first(d: Dict[str, Any], keys: List[str]) -> float:
    for k in keys:
        if k in d:
            try:
                return float(np.asarray(d[k]))
            except Exception:
                pass
    return float("nan")


def load_timescales(path: Path, k: int = 5) -> List[float]:
    if not path.exists():
        return [float("nan")] * k
    x = np.load(path).astype(float).ravel()
    out = list(x[:k])
    if len(out) < k:
        out.extend([float("nan")] * (k - len(out)))
    return out


def load_eigenvalues(path: Path, k: int = 5) -> List[float]:
    if not path.exists():
        return [float("nan")] * k
    x = np.load(path).astype(float).ravel()
    out = list(x[:k])
    if len(out) < k:
        out.extend([float("nan")] * (k - len(out)))
    return out


def relerr(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a == 0.0:
        return float("nan")
    return abs(b - a) / abs(a)


def log10_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return float("nan")
    return math.log10(b / a)


def generator_sanity(Q: csr_matrix, tol: float) -> Dict[str, float]:
    """Check the generator signs and column sums."""
    Qcoo = Q.tocoo()
    diag_mask = (Qcoo.row == Qcoo.col)
    diag = Qcoo.data[diag_mask]
    off = Qcoo.data[~diag_mask]

    diag_pos = int(np.sum(diag > tol))
    off_neg = int(np.sum(off < -tol))


    colsum = np.asarray(Q.sum(axis=0)).ravel()
    max_abs_colsum = float(np.max(np.abs(colsum))) if colsum.size else 0.0


    diag_full = np.asarray(Q.diagonal()).ravel()
    max_exit = float(np.max(np.abs(diag_full))) if diag_full.size else float("nan")
    max_abs_colsum_rel = max_abs_colsum / max_exit if np.isfinite(max_exit) and max_exit > 0 else float("nan")

    return {
        "diag_pos_count": diag_pos,
        "off_neg_count": off_neg,
        "max_abs_colsum": max_abs_colsum,
        "max_abs_colsum_rel": max_abs_colsum_rel,
        "max_abs_diag": max_exit,
    }


def stationarity_metrics(Q: csr_matrix, pi: np.ndarray) -> Tuple[float, float, float]:
    """Return the raw, scaled, and relative stationarity residuals."""
    qpi = Q @ pi
    res = float(np.linalg.norm(np.asarray(qpi).ravel(), 1))

    Qabs = Q.copy()
    Qabs.data = np.abs(Qabs.data)
    scale = float(np.linalg.norm(np.asarray((Qabs @ pi)).ravel(), 1))

    rel = res / scale if scale > 0 else float("nan")
    return res, scale, rel


def graph_metrics_from_Q(Q: csr_matrix) -> Dict[str, float]:
    """Get edge, degree, and component counts from the off-diagonal rates."""

    A = Q.copy().tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    N = A.shape[0]
    edges_dir = int(A.nnz)


    in_deg = np.diff(A.indptr).astype(int)
    in_mean = float(in_deg.mean()) if N else float("nan")
    in_min = int(in_deg.min()) if N else 0
    in_max = int(in_deg.max()) if N else 0
    in_med = float(np.median(in_deg)) if N else float("nan")


    Ac = A.tocsc()
    out_deg = np.diff(Ac.indptr).astype(int)
    out_mean = float(out_deg.mean()) if N else float("nan")
    out_min = int(out_deg.min()) if N else 0
    out_max = int(out_deg.max()) if N else 0
    out_med = float(np.median(out_deg)) if N else float("nan")


    Au = (A + A.T).tocsr()
    Au.data[:] = 1.0
    Au.eliminate_zeros()

    n_comp, labels = connected_components(Au, directed=False, return_labels=True)
    sizes = np.bincount(labels) if labels.size else np.array([0], dtype=int)
    largest = int(sizes.max()) if sizes.size else 0
    largest_frac = float(largest / N) if N else float("nan")


    edges_undir = float(Au.nnz) / 2.0

    return {
        "edges_dir": float(edges_dir),
        "edges_undir": float(edges_undir),
        "in_deg_mean": in_mean,
        "in_deg_median": in_med,
        "in_deg_min": float(in_min),
        "in_deg_max": float(in_max),
        "out_deg_mean": out_mean,
        "out_deg_median": out_med,
        "out_deg_min": float(out_min),
        "out_deg_max": float(out_max),
        "n_components_undirected": float(n_comp),
        "largest_component_frac": largest_frac,
    }


def detect_variant(dps_name: str) -> str:
    name = dps_name.lower()
    if "19sb" in name:
        return "19sb"
    if "99idps" in name:
        return "99idps"
    return "unknown"


def iter_dps_dirs(root: Path, tag: str) -> List[Path]:

    markov_dirs = sorted(root.glob(f"*/*/markov_{tag}"))
    dps_dirs = sorted({p.parent for p in markov_dirs})
    return dps_dirs






def main():
    ap = argparse.ArgumentParser(description="Validate GT coarse-graining via MFPT agreement and extract eigen/graph features.")
    ap.add_argument("--root", type=Path, required=True, help="Root folder, e.g. .../LAMMPS_uncapped")
    ap.add_argument("--T", type=int, default=300, help="Temperature in K (default 300)")
    ap.add_argument("--out-dir", type=Path, default=Path("."), help="Where to write CSV/TXT outputs")
    ap.add_argument("--prefix", type=str, default="", help="Optional prefix for output filenames")
    ap.add_argument("--mfpt-rtol", type=float, default=1e-6, help="Relative tolerance for MFPT match (default 1e-6)")
    ap.add_argument("--stationarity-rtol", type=float, default=1e-10, help="Relative tolerance for stationarity residual (default 1e-10)")
    ap.add_argument("--sign-tol", type=float, default=1e-12, help="Tolerance for sign checks (default 1e-12)")
    args = ap.parse_args()

    root = args.root.resolve()
    tag = parse_tag(args.T)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = (args.prefix + "_") if args.prefix else ""
    csv_path = out_dir / f"{prefix}micro_vs_coarse_{tag}_full.csv"
    txt_path = out_dir / f"{prefix}micro_vs_coarse_{tag}_summary.txt"

    dps_dirs = iter_dps_dirs(root, tag)
    rows: List[Dict[str, Any]] = []

    for dps in dps_dirs:
        markov_dir = dps / f"markov_{tag}"
        eff_dir = markov_dir / f"GT_kept_{tag}"

        system_dir = dps.parent.name
        sequence = system_dir.replace("_nocap", "")
        variant = detect_variant(dps.name)

        row: Dict[str, Any] = {
            "sequence": sequence,
            "system": system_dir,
            "dps_dir": dps.name,
            "variant": variant,
            "T_K": args.T,
            "markov_dir": str(markov_dir),
            "coarse_dir": str(eff_dir),
        }


        micro_npz_path = markov_dir / f"AB_kinetics_{tag}.npz"
        micro_npz = safe_load_npz(micro_npz_path)
        if micro_npz is None:
            row["status"] = "MISSING_MICRO_AB"
            rows.append(row)
            continue

        mfpt_micro_AB = get_first(micro_npz, ["MFPT_AB", "MFPTAB"])
        mfpt_micro_BA = get_first(micro_npz, ["MFPT_BA", "MFPTBA"])
        row["MFPT_micro_AB"] = mfpt_micro_AB
        row["MFPT_micro_BA"] = mfpt_micro_BA
        row["nA_micro"] = get_first(micro_npz, ["nA"])
        row["nB_micro"] = get_first(micro_npz, ["nB"])


        micro_pi_path = markov_dir / f"pi_{tag}.npy"
        if micro_pi_path.exists():
            row["N_micro"] = int(np.load(micro_pi_path).shape[0])
        else:
            row["N_micro"] = ""


        eff_npz_path = eff_dir / f"AB_kinetics_{tag}.npz"
        eff_npz = safe_load_npz(eff_npz_path)
        if eff_npz is None:
            row["status"] = "MISSING_COARSE_AB"
            rows.append(row)
            continue

        mfpt_eff_AB = get_first(eff_npz, ["MFPT_AB", "MFPTAB"])
        mfpt_eff_BA = get_first(eff_npz, ["MFPT_BA", "MFPTBA"])
        row["MFPT_coarse_AB"] = mfpt_eff_AB
        row["MFPT_coarse_BA"] = mfpt_eff_BA
        row["nA_coarse"] = get_first(eff_npz, ["nA"])
        row["nB_coarse"] = get_first(eff_npz, ["nB"])


        pi_eff_path = eff_dir / f"pi_eff_{tag}.npy"
        if pi_eff_path.exists():
            row["N_coarse"] = int(np.load(pi_eff_path).shape[0])
        else:
            row["N_coarse"] = ""


        row["relerr_AB"] = relerr(mfpt_micro_AB, mfpt_eff_AB)
        row["relerr_BA"] = relerr(mfpt_micro_BA, mfpt_eff_BA)
        row["log10_ratio_AB"] = log10_ratio(mfpt_micro_AB, mfpt_eff_AB)
        row["log10_ratio_BA"] = log10_ratio(mfpt_micro_BA, mfpt_eff_BA)


        Q_eff_path = eff_dir / f"Q_eff_{tag}.npz"
        if not Q_eff_path.exists() or not pi_eff_path.exists():
            row["status"] = "MISSING_QEFF_OR_PI"
            rows.append(row)
            continue

        Qeff = load_npz(Q_eff_path).tocsr()
        pi_eff = np.load(pi_eff_path).astype(float).ravel()


        qpi1, absqpi1, qpi_rel = stationarity_metrics(Qeff, pi_eff)
        row["Qpi_norm1"] = qpi1
        row["absQpi_norm1"] = absqpi1
        row["Qpi_rel"] = qpi_rel


        san = generator_sanity(Qeff, tol=args.sign_tol)
        row.update(san)


        row.update(graph_metrics_from_Q(Qeff))


        lam = load_eigenvalues(eff_dir / f"eigenvalues_{tag}.npy", k=5)
        ts = load_timescales(eff_dir / f"timescales_{tag}.npy", k=5)
        for i in range(5):
            row[f"lambda{i+1}"] = lam[i]
            row[f"t{i+1}"] = ts[i]

        if np.isfinite(row["t1"]) and np.isfinite(row["t2"]) and row["t2"] != 0:
            row["t1_over_t2"] = row["t1"] / row["t2"]
        else:
            row["t1_over_t2"] = float("nan")


        mfpt_ok = (
            np.isfinite(row["relerr_AB"]) and row["relerr_AB"] <= args.mfpt_rtol and
            np.isfinite(row["relerr_BA"]) and row["relerr_BA"] <= args.mfpt_rtol
        )
        stationarity_ok = (np.isfinite(qpi_rel) and qpi_rel <= args.stationarity_rtol)
        signs_ok = (row["off_neg_count"] == 0 and row["diag_pos_count"] == 0)
        ab_ok = (row["nA_coarse"] not in ["", float("nan")] and row["nB_coarse"] not in ["", float("nan")] and
                 float(row["nA_coarse"]) >= 1.0 and float(row["nB_coarse"]) >= 1.0)


        conn_ok = (np.isfinite(row["n_components_undirected"]) and row["n_components_undirected"] == 1.0)

        row["mfpt_ok"] = int(mfpt_ok)
        row["stationarity_ok"] = int(stationarity_ok)
        row["signs_ok"] = int(signs_ok)
        row["ab_ok"] = int(ab_ok)
        row["connectivity_ok"] = int(conn_ok)

        row["GT_valid"] = int(mfpt_ok and stationarity_ok and signs_ok and ab_ok and conn_ok)
        row["status"] = "OK" if row["GT_valid"] == 1 else "CHECK"

        rows.append(row)


    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


    def finite_list(vals):
        return [v for v in vals if isinstance(v, (int, float, np.floating)) and np.isfinite(v)]

    ok_rows = [r for r in rows if r.get("status") == "OK"]
    check_rows = [r for r in rows if r.get("status") == "CHECK"]
    missing_rows = [r for r in rows if str(r.get("status","")).startswith("MISSING")]

    relAB = finite_list([r.get("relerr_AB", float("nan")) for r in ok_rows])
    relBA = finite_list([r.get("relerr_BA", float("nan")) for r in ok_rows])

    with txt_path.open("w") as f:
        f.write(f"Micro vs Coarse GT validation report ({tag})\n")
        f.write(f"root: {root}\n")
        f.write(f"total networks scanned: {len(rows)}\n")
        f.write(f"OK: {len(ok_rows)}   CHECK: {len(check_rows)}   MISSING: {len(missing_rows)}\n\n")

        if relAB:
            f.write("MFPT relative error (A->B) over OK rows:\n")
            f.write(f"  min={np.min(relAB):.3e}  median={np.median(relAB):.3e}  max={np.max(relAB):.3e}\n")
        if relBA:
            f.write("MFPT relative error (B->A) over OK rows:\n")
            f.write(f"  min={np.min(relBA):.3e}  median={np.median(relBA):.3e}  max={np.max(relBA):.3e}\n")


        f.write("\nWorst CHECK rows by relerr_AB:\n")
        check_sorted = sorted(check_rows, key=lambda r: (r.get("relerr_AB", float("inf")) if np.isfinite(r.get("relerr_AB", float("nan"))) else float("inf")), reverse=True)
        for r in check_sorted[:10]:
            f.write(f"  {r['system']}/{r['dps_dir']}  "
                    f"relAB={r.get('relerr_AB')}  relBA={r.get('relerr_BA')}  "
                    f"Qpi_rel={r.get('Qpi_rel')}  comps={r.get('n_components_undirected')}  "
                    f"status={r.get('status')}\n")

        if missing_rows:
            f.write("\nMissing cases:\n")
            for r in missing_rows[:50]:
                f.write(f"  {r['system']}/{r['dps_dir']}  status={r['status']}\n")

        f.write("\nWrote:\n")
        f.write(f"  CSV: {csv_path}\n")
        f.write(f"  TXT: {txt_path}\n")

    print(f"Wrote CSV  -> {csv_path}")
    print(f"Wrote TXT  -> {txt_path}")


if __name__ == "__main__":
    main()
