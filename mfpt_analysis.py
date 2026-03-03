#!/usr/bin/env python
"""
mfpt_analysis.py

Compute MFPTs / phenomenological rates between A and B for either
the microscopic KTN or the NGT coarse–grained model, using PyGT.

We assume you have, for each DPS directory and temperature T (e.g. T=300 K):

Microscopic model:
  markov_T{T}K/
      B_T{T}K.npz                  # branching probabilities (sparse)
      Q_T{T}K.npz                  # CTMC generator (sparse, rows sum to 0)
      tau_T{T}K.npy                # waiting times tau_j
      pi_T{T}K.npy                 # stationary distribution pi_j
      orig_min_ids_T{T}K.npy       # or original_min_ids_T{T}K.npy (original PATHSAMPLE IDs)

Coarse NGT model (GT_kept):
  markov_T{T}K/GT_kept_T{T}K/
      B_eff_T{T}K.npz              # effective branching probabilities
      Q_eff_T{T}K.npz              # effective CTMC generator
      tau_eff_T{T}K.npy            # effective waiting times
      pi_eff_T{T}K.npy             # effective stationary distribution
      orig_min_ids_eff_T{T}K.npy   # or original_min_ids_eff_T{T}K.npy

AB selectors (from PATHSAMPLE):
  min.A                            # text file of original minimum IDs in A
  min.B                            # text file of original minimum IDs in B

Outputs, per T and per model:

Microscopic:
  markov_T{T}K/
      eigenvalues_T{T}K.npy
      timescales_T{T}K.npy
      AB_kinetics_T{T}K.npz        # MFPTs and (if applicable) phenomenological rates

Coarse:
  markov_T{T}K/GT_kept_T{T}K/
      eigenvalues_T{T}K.npy
      timescales_T{T}K.npy
      AB_kinetics_T{T}K.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, diags
from scipy.sparse.linalg import eigsh

from PyGT import stats  # PyGT.stats


# ----------------------------------------------------------------------
#  Helpers to load Markov models
# ----------------------------------------------------------------------

def _find_first_existing(candidates):
    """Return the first Path in `candidates` that exists, or None."""
    for p in candidates:
        if p.exists():
            return p
    return None


def load_markov_model(dps_dir: Path, T: float, coarse: bool):
    """
    Load B, tau, Q, pi, orig_min_ids, base_dir, tag for a given sequence and T.

    dps_dir: directory like .../yyggyy_99idps_nocap
    T: temperature in K
    coarse: if True, use GT_kept_T{T}K/ effective model
    """
    tag = f"T{int(round(T))}K"

    if coarse:
        base = dps_dir / f"markov_{tag}" / f"GT_kept_{tag}"
        B_file = base / f"B_eff_{tag}.npz"
        Q_file = base / f"Q_eff_{tag}.npz"
        tau_file = base / f"tau_eff_{tag}.npy"
        pi_file = base / f"pi_eff_{tag}.npy"
        orig_candidates = [
            base / f"orig_min_ids_eff_{tag}.npy",
            base / f"original_min_ids_eff_{tag}.npy",
        ]
    else:
        base = dps_dir / f"markov_{tag}"
        B_file = base / f"B_{tag}.npz"
        Q_file = base / f"Q_{tag}.npz"
        tau_file = base / f"tau_{tag}.npy"
        pi_file = base / f"pi_{tag}.npy"
        orig_candidates = [
            base / f"orig_min_ids_{tag}.npy",
            base / f"original_min_ids_{tag}.npy",
        ]

    if not base.exists():
        raise FileNotFoundError(f"[mfpt_analysis] Markov directory not found: {base}")

    orig_file = _find_first_existing(orig_candidates)
    if orig_file is None:
        raise FileNotFoundError(
            f"[mfpt_analysis] Could not find any orig_min_ids file in {base}.\n"
            f"Tried: {[str(c) for c in orig_candidates]}"
        )

    B = load_npz(B_file)
    Q = load_npz(Q_file)
    tau = np.load(tau_file)
    pi = np.load(pi_file)
    orig_ids = np.load(orig_file)

    return B, tau, Q, pi, orig_ids, base, tag


# ----------------------------------------------------------------------
#  A/B set handling (min.A / min.B)
# ----------------------------------------------------------------------

def _read_min_set(path: Path) -> np.ndarray:
    """
    Read PATHSAMPLE-style min.A / min.B files.

    These are plain text. Common formats:

      - First line = count, then that many integer IDs (possibly multiple per line).
      - Or simply a list of integer IDs, one per line.

    We detect the "count" convention by checking if the first integer equals
    (total_count - 1). If yes, we drop it; otherwise we treat all integers as IDs.
    """
    if not path.exists():
        return np.array([], dtype=int)

    data = np.loadtxt(path, dtype=int, ndmin=1)
    data = np.atleast_1d(data).ravel()

    if data.size == 0:
        return data

    # Heuristic: first integer may be "how many IDs"
    first = int(data[0])
    n_rest = data.size - 1
    if n_rest == first:
        # PATHSAMPLE convention: first entry is count
        return data[1:]
    else:
        # Treat all entries as IDs
        return data


def make_AB_selectors(dps_dir: Path, orig_ids: np.ndarray):
    """
    Build boolean selectors A_sel, B_sel on the *current* model indices,
    using original minimum IDs matched against min.A and min.B in the DPS dir.

    orig_ids: array of original PATHSAMPLE min indices corresponding to
              rows/cols of Q, B, etc. (micro or coarse).
    """
    A_ids = _read_min_set(dps_dir / "min.A")
    B_ids = _read_min_set(dps_dir / "min.B")

    if A_ids.size == 0 or B_ids.size == 0:
        print(f"[mfpt_analysis] WARNING: min.A or min.B empty/missing in {dps_dir}")
        return None, None

    A_set = set(int(x) for x in A_ids)
    B_set = set(int(x) for x in B_ids)

    A_sel = np.array([int(i) in A_set for i in orig_ids], dtype=bool)
    B_sel = np.array([int(i) in B_set for i in orig_ids], dtype=bool)

    return A_sel, B_sel


# ----------------------------------------------------------------------
#  Spectrum: reversible CTMC → symmetric similarity transform
# ----------------------------------------------------------------------

def compute_spectrum(Q, pi, max_eigs: int, out_dir: Path, tag: str):
    """
    Robust slow-mode spectrum for reversible CTMC.

    Computes the slowest nonzero eigenvalues (closest to 0 from below) and
    relaxation timescales tau_k = -1/lambda_k.

    Works best for coarse models (N ~ 10^2–10^4).
    """
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence
    from scipy.sparse.linalg import norm as spnorm

    N = Q.shape[0]
    if max_eigs <= 0 or N <= 1:
        return

    # Ask for one extra eigenvalue to include the stationary mode (0)
    k = min(max_eigs + 1, N - 1)
    if k <= 0:
        return

    # Guard against tiny pi
    pi_safe = np.clip(np.asarray(pi, dtype=float), 1e-300, None)
    sqrt_pi = np.sqrt(pi_safe)
    inv_sqrt_pi = 1.0 / sqrt_pi

    S = diags(sqrt_pi)
    Sinv = diags(inv_sqrt_pi)

    # Reversible symmetric similarity transform of Q^T
    L = S @ Q.T @ Sinv

    # If numerical noise breaks symmetry slightly, symmetrize
    try:
        asym = spnorm(L - L.T, ord=1) / max(spnorm(L, ord=1), 1e-300)
        if asym < 1e-10:
            L = 0.5 * (L + L.T)
    except Exception:
        pass

    def postprocess(vals):
        vals = np.real(vals)
        vals = np.sort(vals)[::-1]  # closest to 0 first
        nonzero = vals[vals < -1e-12]  # drop stationary ~0 mode
        if nonzero.size == 0:
            raise RuntimeError("No nonzero negative eigenvalues found.")
        slow = nonzero[:max_eigs]
        timescales = -1.0 / slow
        np.save(out_dir / f"eigenvalues_{tag}.npy", slow)
        np.save(out_dir / f"timescales_{tag}.npy", timescales)
        print("[mfpt_analysis] slow eigenvalues:", slow[:5])
        print("[mfpt_analysis] timescales:", timescales[:5])

    print(f"[mfpt_analysis] Computing {k-1} slow eigenvalues for N={N}...")

    # 1) Try standard eigsh first
    try:
        vals, _ = eigsh(L, k=k, which="LA", tol=1e-10, maxiter=200000)
        postprocess(vals)
        return
    except ArpackNoConvergence as e:
        print(f"[mfpt_analysis] WARNING: LA eigsh did not converge ({e}); trying shift-invert...")
    except Exception as e:
        print(f"[mfpt_analysis] WARNING: LA eigsh failed ({type(e).__name__}: {e}); trying shift-invert...")

    # 2) Shift-invert near 0 (avoid sigma=0 exactly because of the stationary mode)
    for sigma in (-1e-12, -1e-10, -1e-8):
        try:
            print(f"[mfpt_analysis] Shift-invert eigsh with sigma={sigma} ...")
            vals, _ = eigsh(
                L, k=k,
                sigma=sigma, which="LM",   # eigenvalues closest to sigma become largest in transformed operator
                tol=1e-10, maxiter=500000
            )
            postprocess(vals)
            return
        except ArpackNoConvergence as e:
            print(f"[mfpt_analysis] WARNING: shift-invert sigma={sigma} no convergence ({e}).")
        except Exception as e:
            print(f"[mfpt_analysis] WARNING: shift-invert sigma={sigma} failed ({type(e).__name__}: {e}).")

    print("[mfpt_analysis] WARNING: spectrum failed for this model; skipping.")



# ----------------------------------------------------------------------
#  AB kinetics using PyGT
# ----------------------------------------------------------------------

def compute_AB_kinetics(
    dps_dir: Path,
    B,
    tau,
    Q,
    pi,
    orig_ids,
    out_dir: Path,
    tag: str,
):
    """
    Use PyGT to get MFPTs / rates between A and B and save them.

    Cases:
      - If |A|=0 or |B|=0: just save sizes and bail.
      - If |A|=|B|=1: use PyGT.stats.compute_passage_stats (exact MFPT, no GT).
      - Otherwise: use PyGT.stats.compute_rates, which does graph transformation
        internally and gives MFPTs and phenomenological rates (kSS, kNSS, k*, kF, etc.).
    """
    A_sel, B_sel = make_AB_selectors(dps_dir, orig_ids)
    if A_sel is None or B_sel is None:
        print("[mfpt_analysis] Skipping AB kinetics (no valid A/B sets).")
        return

    nA, nB = int(A_sel.sum()), int(B_sel.sum())
    print(f"[mfpt_analysis] |A| = {nA}, |B| = {nB}")

    results = {"nA": nA, "nB": nB}

    if nA == 0 or nB == 0:
        print("[mfpt_analysis] Either A or B is empty; not computing MFPTs.")
        np.savez(out_dir / f"AB_kinetics_{tag}.npz", **results)
        return

    # Degenerate case: exactly one source microstate and one sink microstate.
    # PyGT.stats.compute_rates explicitly rejects this case, so we use the
    # direct MFPT formula from compute_passage_stats instead. 
    if nA == 1 and nB == 1:
        print(
            "[mfpt_analysis] A and B each have 1 state; "
            "using PyGT.stats.compute_passage_stats for MFPTs only."
        )
        # With dopdf=False, compute_passage_stats returns only tau:
        # tau = [T_BA, Var_BA, T_AB, Var_AB]. 
        tau_moments = stats.compute_passage_stats(
            A_sel, B_sel, pi, Q, dopdf=False
        )

        results.update(
            {
                "MFPT_BA": float(tau_moments[0]),
                "Var_BA": float(tau_moments[1]),
                "MFPT_AB": float(tau_moments[2]),
                "Var_AB": float(tau_moments[3]),
            }
        )
    else:
        # General case: use Wales/Swinburne phenomenological rates.
        # compute_rates() removes intermediates (I set) by GT, then solves
        # the MFPT and rate problem on the reduced A∪B network. 
        print("[mfpt_analysis] Using PyGT.stats.compute_rates for MFPTs and rates...")
        rate_dict = stats.compute_rates(
            A_sel,
            B_sel,
            B,
            tau,
            pi,
            MFPTonly=False,
            fullGT=False,
            screen=False,
        )
        # Flatten dictionary to plain Python/scalar types
        for k, v in rate_dict.items():
            results[k] = float(np.asarray(v))

    out_path = out_dir / f"AB_kinetics_{tag}.npz"
    np.savez(out_path, **results)
    print(f"[mfpt_analysis] Saved AB kinetics → {out_path}")


# ----------------------------------------------------------------------
#  Main driver
# ----------------------------------------------------------------------

def analyse_one(dps_dir: Path, T: float, coarse: bool, max_eigs: int):
    """
    Run spectrum + AB kinetics for a single DPS directory at temperature T.
    """
    model_label = "coarse" if coarse else "micro"
    print(f"[mfpt_analysis] Analysing {dps_dir} at T = {T} K ({model_label} model)")

    B, tau, Q, pi, orig_ids, out_dir, tag = load_markov_model(dps_dir, T, coarse)
    N = Q.shape[0]
    print(f"[mfpt_analysis] Model: {model_label}, N = {N}")

    # 1) Spectrum (slowest relaxation modes)
    compute_spectrum(Q, pi=pi, max_eigs=max_eigs, out_dir=out_dir, tag=tag)

    # 2) A/B MFPTs and phenomenological rates
    compute_AB_kinetics(
        dps_dir=dps_dir,
        B=B,
        tau=tau,
        Q=Q,
        pi=pi,
        orig_ids=orig_ids,
        out_dir=out_dir,
        tag=tag,
    )


def main():
    parser = argparse.ArgumentParser(
        description="MFPT / rate analysis using PyGT for one DPS directory."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to DPS directory (e.g. .../yyggyy_99idps_nocap)",
    )
    parser.add_argument(
        "--T",
        type=float,
        required=True,
        help="Temperature in K (matches markov_T{T}K folders).",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="Use NGT coarse–grained model (markov_T{T}K/GT_kept_T{T}K).",
    )
    parser.add_argument(
        "--max-eigs",
        type=int,
        default=10,
        help="Number of slow modes (non-zero eigenvalues of Q) to compute.",
    )

    args = parser.parse_args()
    analyse_one(args.data_dir.resolve(), args.T, coarse=args.coarse, max_eigs=args.max_eigs)


if __name__ == "__main__":
    main()
