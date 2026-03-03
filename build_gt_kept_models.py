#!/usr/bin/env python3
"""
build_gt_kept_models.py

Create GT-eliminated ("kept") coarse-grained CTMC models from microscopic KTN data.

For each markov_T{T}K/ directory, we:
  1) Load B, tau, pi, and original_min_ids (micro model)
  2) Build A/B selectors from min.A/min.B (always retained)
  3) Choose additional kept minima using a percentile-based criterion
     (recommended: style='hybrid' using both free energy -log(pi) and escape time tau)
  4) Apply NGT / graph transformation to eliminate all other nodes:
        B_eff, tau_eff, Q_eff = PyGT.GT.blockGT(rm_vec, B, tau, rates=True, ...)
  5) Save coarse model to: markov_T{T}K/GT_kept_T{T}K/

Outputs per markov_T{T}K/:
  GT_kept_T{T}K/
      B_eff_T{T}K.npz
      Q_eff_T{T}K.npz
      tau_eff_T{T}K.npy
      pi_eff_T{T}K.npy
      original_min_ids_eff_T{T}K.npy
      kept_mask_eff_T{T}K.npy
      micro_to_eff_index_T{T}K.npy
      A_states_T{T}K.npy
      B_states_T{T}K.npy

Notes:
- Uses column-sum convention for Q.
- pi_eff is taken as pi restricted to kept states and renormalized, then checked
  against Q_eff; if the stationarity residual is poor, we fall back to solving Q_eff pi=0.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, diags, isspmatrix, load_npz, save_npz
from scipy.sparse.linalg import spsolve

import PyGT.GT as GT


# -------------------------
# Utilities: file discovery
# -------------------------
TAG_RE = re.compile(r"^markov_(T\d+K)$")


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def parse_tag(markov_dir: Path) -> Optional[str]:
    m = TAG_RE.match(markov_dir.name)
    return m.group(1) if m else None


def iter_markov_dirs(root: Path, only_T: Optional[int]) -> Iterable[Path]:
    # Expected: root/SEQ_DIR/DPS_DIR/markov_TxxxK
    for md in root.glob("*/*/markov_T*"):
        if not md.is_dir():
            continue
        tag = parse_tag(md)
        if tag is None:
            continue
        if only_T is not None:
            want = f"T{only_T}K"
            if tag != want:
                continue
        yield md


# -------------------------
# A/B set handling
# -------------------------
def _read_min_set(path: Path) -> np.ndarray:
    """
    Read PATHSAMPLE min.A/min.B:
      - either: first entry is count, followed by that many IDs
      - or: plain list of IDs
    """
    if not path.exists():
        return np.array([], dtype=int)
    data = np.loadtxt(path, dtype=int, ndmin=1)
    data = np.atleast_1d(data).ravel()
    if data.size == 0:
        return data
    first = int(data[0])
    if data.size - 1 == first:
        return data[1:]
    return data


def make_AB_selectors(dps_dir: Path, orig_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A_ids = _read_min_set(dps_dir / "min.A")
    B_ids = _read_min_set(dps_dir / "min.B")

    A_set = set(int(x) for x in A_ids.tolist())
    B_set = set(int(x) for x in B_ids.tolist())

    A_sel = np.array([int(i) in A_set for i in orig_ids], dtype=bool)
    B_sel = np.array([int(i) in B_set for i in orig_ids], dtype=bool)
    return A_sel, B_sel


# -------------------------
# Selection criteria (kept minima)
# -------------------------
def choose_rm_vec(
    pi: np.ndarray,
    tau: np.ndarray,
    must_keep: np.ndarray,
    style: str,
    percent_retained: float,
    min_kept: int,
) -> np.ndarray:
    """
    Return rm_vec (True = remove) using percentile rules similar to PyGT.tools.choose_nodes_to_remove,
    but implemented here explicitly (and without the node_degree signature bug).

    rm_region = nodes we are allowed to remove (i.e. not must_keep).
    """
    if pi.ndim != 1 or tau.ndim != 1:
        raise ValueError("pi and tau must be 1D arrays.")
    if pi.shape[0] != tau.shape[0] or pi.shape[0] != must_keep.shape[0]:
        raise ValueError("pi, tau, must_keep must have the same length.")

    N = pi.size
    rm_region = ~must_keep

    # If nothing is removable, rm_vec is all False.
    if not rm_region.any():
        return np.zeros(N, dtype=bool)

    # Clip pi to avoid log underflow
    pi_safe = np.clip(pi.astype(float), 1e-300, None)
    tau_safe = tau.astype(float)

    style = style.lower().strip()
    pr = float(percent_retained)

    if not (0.0 < pr <= 100.0):
        raise ValueError("--percent-retained must be in (0, 100].")

    rm_vec = np.zeros(N, dtype=bool)

    if style == "free_energy":
        fe = -np.log(pi_safe)
        # keep lowest fe; remove fe above the pr-th percentile among removable
        thresh = np.percentile(fe[rm_region], pr)
        rm_vec[rm_region] = fe[rm_region] > thresh

    elif style == "escape_time":
        # keep slowest tau; remove tau below the (100-pr)-th percentile among removable
        thresh = np.percentile(tau_safe[rm_region], 100.0 - pr)
        rm_vec[rm_region] = tau_safe[rm_region] < thresh

    elif style == "combined":
        # keep largest tau*pi (slow & populated); remove below (100-pr)-th percentile
        metric = tau_safe * pi_safe
        thresh = np.percentile(metric[rm_region], 100.0 - pr)
        rm_vec[rm_region] = metric[rm_region] < thresh

    elif style == "hybrid":
        # remove only those that are BOTH fast and high free energy
        fe = -np.log(pi_safe)
        fe_thresh = np.percentile(fe[rm_region], pr)                # high FE = bad
        tau_thresh = np.percentile(tau_safe[rm_region], 100.0 - pr)  # low tau = fast
        rm_vec[rm_region] = (fe[rm_region] > fe_thresh) & (tau_safe[rm_region] < tau_thresh)

    else:
        raise ValueError("Invalid --style. Use: free_energy, escape_time, combined, hybrid.")

    # Enforce a minimum number of kept nodes (including must_keep)
    keep = ~rm_vec
    if keep.sum() < min_kept:
        # Add extra nodes by highest pi among removable nodes until min_kept is reached
        candidates = np.where(rm_region & rm_vec)[0]  # removable but currently removed
        if candidates.size > 0:
            order = candidates[np.argsort(pi_safe[candidates])[::-1]]
            need = min_kept - int(keep.sum())
            add = order[:need]
            rm_vec[add] = False

    # Never remove must_keep
    rm_vec[must_keep] = False
    return rm_vec


# -------------------------
# Stationary distribution on reduced model
# -------------------------
def stationarity_residual(Q: csr_matrix, pi: np.ndarray) -> Tuple[float, float, float]:
    """
    Returns (||Q pi||_1, || |Q| pi ||_1, relative).
    """
    r = np.linalg.norm((Q @ pi), 1)
    s = np.linalg.norm((abs(Q) @ pi), 1)
    rel = (r / s) if s > 0 else np.nan
    return float(r), float(s), float(rel)


def solve_stationary(Q: csr_matrix) -> np.ndarray:
    """
    Solve Q pi = 0 with sum(pi)=1 by replacing the first row with ones.
    """
    N = Q.shape[0]
    A = Q.tolil()
    b = np.zeros(N, dtype=float)
    A[0, :] = 1.0
    b[0] = 1.0
    pi = spsolve(A.tocsr(), b).astype(float)
    # Clean up numerical noise
    pi[pi < 0] = 0.0
    s = pi.sum()
    if s <= 0:
        raise RuntimeError("Failed to compute stationary distribution (sum <= 0).")
    return pi / s


# -------------------------
# Main per-directory routine
# -------------------------
@dataclass
class BuildResult:
    status: str
    markov_dir: Path
    out_dir: Path
    tag: str
    N: int
    N_eff: int
    nA: int
    nB: int
    rel_stationarity_eff: float


def build_one(markov_dir: Path, style: str, percent_retained: float, min_kept: int,
              block: int, cond_thresh: float, screen: bool, overwrite: bool) -> BuildResult:
    tag = parse_tag(markov_dir)
    if tag is None:
        raise ValueError(f"Not a markov_T*K dir: {markov_dir}")

    dps_dir = markov_dir.parent
    out_dir = markov_dir / f"GT_kept_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Output guard
    q_eff_path = out_dir / f"Q_eff_{tag}.npz"
    if q_eff_path.exists() and not overwrite:
        return BuildResult(
            status="SKIP_EXISTS",
            markov_dir=markov_dir,
            out_dir=out_dir,
            tag=tag,
            N=-1, N_eff=-1, nA=-1, nB=-1,
            rel_stationarity_eff=np.nan,
        )

    # Load required inputs
    B_path = markov_dir / f"B_{tag}.npz"
    tau_path = markov_dir / f"tau_{tag}.npy"
    pi_path = markov_dir / f"pi_{tag}.npy"

    orig_path = _first_existing([
        markov_dir / f"original_min_ids_{tag}.npy",
        markov_dir / f"orig_min_ids_{tag}.npy",
    ])

    if orig_path is None:
        raise FileNotFoundError(f"Missing original_min_ids/orig_min_ids for {markov_dir}")

    if not (tau_path.exists() and pi_path.exists()):
        raise FileNotFoundError(f"Missing tau/pi for {markov_dir}")

    tau = np.load(tau_path).astype(float)
    pi = np.load(pi_path).astype(float)
    orig_ids = np.load(orig_path)

    N = tau.size
    if pi.size != N:
        raise ValueError(f"pi length {pi.size} != tau length {N} in {markov_dir}")

    # Load branching matrix; if missing, attempt to build from Q
    if B_path.exists():
        B = load_npz(B_path).tocsr()
    else:
        Q_path = markov_dir / f"Q_{tag}.npz"
        if not Q_path.exists():
            raise FileNotFoundError(f"Missing B and Q in {markov_dir}")
        Q = load_npz(Q_path).tocsr()
        # B = offdiag(Q) * diag(tau)
        Q_off = Q - diags(Q.diagonal())
        B = (Q_off @ diags(tau)).tocsr()

    # Must-keep set = A ∪ B (if available)
    A_sel, B_sel = make_AB_selectors(dps_dir, orig_ids)
    nA, nB = int(A_sel.sum()), int(B_sel.sum())
    must_keep = (A_sel | B_sel)

    # If min.A/min.B missing, must_keep will be empty; still proceed
    rm_vec = choose_rm_vec(
        pi=pi, tau=tau, must_keep=must_keep,
        style=style, percent_retained=percent_retained,
        min_kept=min_kept,
    )
    keep_mask = ~rm_vec
    N_eff_target = int(keep_mask.sum())
    if N_eff_target < 2:
        # Force at least 2 nodes retained: keep top-2 by pi
        top2 = np.argsort(pi)[::-1][:2]
        keep_mask[:] = False
        keep_mask[top2] = True
        keep_mask[must_keep] = True
        rm_vec = ~keep_mask
        N_eff_target = int(keep_mask.sum())

    # Run GT (this is the expensive step)
    # blockGT returns (B_eff, tau_eff, K_eff) where K_eff is the GT-reduced generator (column-sum convention)
    B_eff, tau_eff, Q_eff = GT.blockGT(
        rm_vec=rm_vec,
        B=B,
        tau=tau,
        block=int(block),
        order=None,
        rates=True,
        screen=bool(screen),
        cond_thresh=float(cond_thresh),
    )

    # Normalize output types
    if not isspmatrix(B_eff):
        B_eff = csr_matrix(B_eff)
    else:
        B_eff = B_eff.tocsr()

    if not isspmatrix(Q_eff):
        Q_eff = csr_matrix(Q_eff)
    else:
        Q_eff = Q_eff.tocsr()

    tau_eff = np.asarray(tau_eff, dtype=float).ravel()

    # Build mapping arrays
    keep_idx = np.where(keep_mask)[0]
    orig_ids_eff = orig_ids[keep_mask]

    micro_to_eff = -np.ones(N, dtype=int)
    micro_to_eff[keep_idx] = np.arange(keep_idx.size, dtype=int)

    # Reduced A/B selectors
    A_eff = A_sel[keep_mask]
    B_eff_sel = B_sel[keep_mask]

    # pi_eff: start with restricted pi; fall back to solving if needed
    pi_eff = np.asarray(pi[keep_mask], dtype=float)
    pi_eff_sum = pi_eff.sum()
    if pi_eff_sum <= 0:
        pi_eff = solve_stationary(Q_eff)
    else:
        pi_eff /= pi_eff_sum
        _, _, rel = stationarity_residual(Q_eff, pi_eff)
        if not np.isfinite(rel) or rel > 1e-10:
            pi_eff = solve_stationary(Q_eff)

    # Final stationarity check
    _, _, rel_eff = stationarity_residual(Q_eff, pi_eff)

    # Save outputs
    save_npz(out_dir / f"B_eff_{tag}.npz", B_eff)
    save_npz(out_dir / f"Q_eff_{tag}.npz", Q_eff)
    np.save(out_dir / f"tau_eff_{tag}.npy", tau_eff)
    np.save(out_dir / f"pi_eff_{tag}.npy", pi_eff)
    np.save(out_dir / f"original_min_ids_eff_{tag}.npy", orig_ids_eff)
    np.save(out_dir / f"kept_mask_eff_{tag}.npy", keep_mask)
    np.save(out_dir / f"micro_to_eff_index_{tag}.npy", micro_to_eff)
    np.save(out_dir / f"A_states_{tag}.npy", A_eff)
    np.save(out_dir / f"B_states_{tag}.npy", B_eff_sel)

    return BuildResult(
        status="OK",
        markov_dir=markov_dir,
        out_dir=out_dir,
        tag=tag,
        N=N,
        N_eff=int(Q_eff.shape[0]),
        nA=nA,
        nB=nB,
        rel_stationarity_eff=float(rel_eff),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build GT_kept coarse models using PyGT graph transformation.")
    ap.add_argument("--root", type=Path, required=True, help="Root folder, e.g. .../LAMMPS_uncapped")
    ap.add_argument("--T", type=int, default=None, help="Only process markov_T{T}K (e.g. 300). Default: all.")
    ap.add_argument("--style", type=str, default="hybrid",
                    choices=["free_energy", "escape_time", "combined", "hybrid"],
                    help="Criterion to decide what to remove. Recommended: hybrid.")
    ap.add_argument("--percent-retained", type=float, default=5.0,
                    help="Percentile parameter (interpreted as in PyGT docs). Try 1,2,5,10.")
    ap.add_argument("--min-kept", type=int, default=50,
                    help="Guarantee at least this many nodes are retained (including A/B).")
    ap.add_argument("--block", type=int, default=20, help="PyGT block size for blockGT.")
    ap.add_argument("--cond-thresh", type=float, default=1e13, help="Condition threshold for block inversion.")
    ap.add_argument("--screen", action="store_true", help="Print PyGT progress.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing GT_kept outputs.")
    ap.add_argument("--report", type=Path, default=Path("gt_kept_build_report.txt"),
                    help="Write a plain-text summary report here.")
    args = ap.parse_args()

    root = args.root.resolve()
    results: list[BuildResult] = []

    for md in iter_markov_dirs(root, args.T):
        try:
            res = build_one(
                md,
                style=args.style,
                percent_retained=args.percent_retained,
                min_kept=args.min_kept,
                block=args.block,
                cond_thresh=args.cond_thresh,
                screen=args.screen,
                overwrite=args.overwrite,
            )
        except Exception as e:
            tag = parse_tag(md) or "UNKNOWN"
            out_dir = md / f"GT_kept_{tag}"
            results.append(BuildResult(
                status=f"FAIL: {type(e).__name__}",
                markov_dir=md,
                out_dir=out_dir,
                tag=tag,
                N=-1, N_eff=-1, nA=-1, nB=-1,
                rel_stationarity_eff=np.nan,
            ))
            print(f"[FAIL] {md}  ({type(e).__name__}: {e})", flush=True)
            continue

        print(f"[{res.status}] {md}  N={res.N} -> N_eff={res.N_eff}  rel_stationarity_eff={res.rel_stationarity_eff:.2e}",
              flush=True)
        results.append(res)

    # Write report
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write(f"GT-kept build report for root: {root}\n")
        fh.write(f"style={args.style}  percent_retained={args.percent_retained}  min_kept={args.min_kept}\n")
        fh.write(f"block={args.block}  cond_thresh={args.cond_thresh}\n\n")
        for r in results:
            fh.write("=" * 90 + "\n")
            fh.write(f"{r.markov_dir}\n")
            fh.write(f"status: {r.status}\n")
            if r.status == "OK":
                fh.write(f"tag: {r.tag}\n")
                fh.write(f"N: {r.N}\n")
                fh.write(f"N_eff: {r.N_eff}\n")
                fh.write(f"nA: {r.nA}   nB: {r.nB}\n")
                fh.write(f"rel_stationarity_eff: {r.rel_stationarity_eff:.3e}\n")
                fh.write(f"out_dir: {r.out_dir}\n")
        fh.write("\nDone.\n")

    print(f"\n[build_gt_kept_models] Wrote report -> {args.report}")


if __name__ == "__main__":
    main()
