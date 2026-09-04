#!/usr/bin/env python
"""Compute slow modes and A/B passage times with PyGT.

Works with either the microscopic model or its ``GT_kept`` version and saves
the arrays next to that model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, diags
from scipy.sparse.linalg import eigsh

from PyGT import stats






def _find_first_existing(candidates):
    """Take the first candidate that exists."""
    for p in candidates:
        if p.exists():
            return p
    return None


def load_markov_model(dps_dir: Path, T: float, coarse: bool):
    """Load the arrays for one temperature, optionally from ``GT_kept``."""
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






def _read_min_set(path: Path) -> np.ndarray:
    """Read min.A/min.B, dropping the optional leading count."""
    if not path.exists():
        return np.array([], dtype=int)

    data = np.loadtxt(path, dtype=int, ndmin=1)
    data = np.atleast_1d(data).ravel()

    if data.size == 0:
        return data


    first = int(data[0])
    n_rest = data.size - 1
    if n_rest == first:

        return data[1:]
    else:

        return data


def make_AB_selectors(dps_dir: Path, orig_ids: np.ndarray):
    """Map the original min.A/min.B IDs into the current model indices."""
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






def compute_spectrum(Q, pi, max_eigs: int, out_dir: Path, tag: str):
    """Get the slow nonzero CTMC modes and their relaxation times."""
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import eigsh, ArpackNoConvergence
    from scipy.sparse.linalg import norm as spnorm

    N = Q.shape[0]
    if max_eigs <= 0 or N <= 1:
        return


    k = min(max_eigs + 1, N - 1)
    if k <= 0:
        return


    pi_safe = np.clip(np.asarray(pi, dtype=float), 1e-300, None)
    sqrt_pi = np.sqrt(pi_safe)
    inv_sqrt_pi = 1.0 / sqrt_pi

    S = diags(sqrt_pi)
    Sinv = diags(inv_sqrt_pi)


    L = S @ Q.T @ Sinv


    try:
        asym = spnorm(L - L.T, ord=1) / max(spnorm(L, ord=1), 1e-300)
        if asym < 1e-10:
            L = 0.5 * (L + L.T)
    except Exception:
        pass

    def postprocess(vals):
        vals = np.real(vals)
        vals = np.sort(vals)[::-1]
        nonzero = vals[vals < -1e-12]
        if nonzero.size == 0:
            raise RuntimeError("No nonzero negative eigenvalues found.")
        slow = nonzero[:max_eigs]
        timescales = -1.0 / slow
        np.save(out_dir / f"eigenvalues_{tag}.npy", slow)
        np.save(out_dir / f"timescales_{tag}.npy", timescales)
        print("[mfpt_analysis] slow eigenvalues:", slow[:5])
        print("[mfpt_analysis] timescales:", timescales[:5])

    print(f"[mfpt_analysis] Computing {k-1} slow eigenvalues for N={N}...")


    try:
        vals, _ = eigsh(L, k=k, which="LA", tol=1e-10, maxiter=200000)
        postprocess(vals)
        return
    except ArpackNoConvergence as e:
        print(f"[mfpt_analysis] WARNING: LA eigsh did not converge ({e}); trying shift-invert...")
    except Exception as e:
        print(f"[mfpt_analysis] WARNING: LA eigsh failed ({type(e).__name__}: {e}); trying shift-invert...")


    for sigma in (-1e-12, -1e-10, -1e-8):
        try:
            print(f"[mfpt_analysis] Shift-invert eigsh with sigma={sigma} ...")
            vals, _ = eigsh(
                L, k=k,
                sigma=sigma, which="LM",
                tol=1e-10, maxiter=500000
            )
            postprocess(vals)
            return
        except ArpackNoConvergence as e:
            print(f"[mfpt_analysis] WARNING: shift-invert sigma={sigma} no convergence ({e}).")
        except Exception as e:
            print(f"[mfpt_analysis] WARNING: shift-invert sigma={sigma} failed ({type(e).__name__}: {e}).")

    print("[mfpt_analysis] WARNING: spectrum failed for this model; skipping.")







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
    """Calculate and save the A/B MFPTs.

    Singleton A/B sets use ``compute_passage_stats``; larger sets go through
    PyGT's graph-transformation rate calculation.
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




    if nA == 1 and nB == 1:
        print(
            "[mfpt_analysis] A and B each have 1 state; "
            "using PyGT.stats.compute_passage_stats for MFPTs only."
        )


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

        for k, v in rate_dict.items():
            results[k] = float(np.asarray(v))

    out_path = out_dir / f"AB_kinetics_{tag}.npz"
    np.savez(out_path, **results)
    print(f"[mfpt_analysis] Saved AB kinetics → {out_path}")






def analyse_one(dps_dir: Path, T: float, coarse: bool, max_eigs: int):
    """Run the spectrum and A/B calculation for one DPS folder."""
    model_label = "coarse" if coarse else "micro"
    print(f"[mfpt_analysis] Analysing {dps_dir} at T = {T} K ({model_label} model)")

    B, tau, Q, pi, orig_ids, out_dir, tag = load_markov_model(dps_dir, T, coarse)
    N = Q.shape[0]
    print(f"[mfpt_analysis] Model: {model_label}, N = {N}")


    compute_spectrum(Q, pi=pi, max_eigs=max_eigs, out_dir=out_dir, tag=tag)


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
