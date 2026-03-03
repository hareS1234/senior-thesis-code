import numpy as np
from typing import Iterable, Dict, Any


def compute_mfpt_from_Q(
    Q: np.ndarray,
    A: Iterable[int],
    B: Iterable[int],
) -> float:
    """
    Compute mean first-passage time (MFPT) from set A to set B
    for a continuous-time Markov chain with generator Q.

    Assumes:
      - Q is *row* generator: rows sum to 0 and p(t) evolves as p^T Q.
      - A and B are sets of integer state indices (0-based).

    If your Q is *column*-stochastic (as in your thesis text), pass Q.T here,
    or build a row-generator version before calling this function.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    A = np.array(sorted(set(A)), dtype=int)
    B = np.array(sorted(set(B)), dtype=int)

    all_states = np.arange(n, dtype=int)
    # states that are NOT absorbing (i.e. not in B)
    mask_notB = np.ones(n, dtype=bool)
    mask_notB[B] = False
    I = all_states[mask_notB]

    # Submatrix Q_II, and RHS -1 for all i in I
    Q_II = Q[np.ix_(I, I)]
    b = -np.ones(len(I))

    # Solve for expected hitting times m_i to B
    m = np.zeros(n)
    m_I = np.linalg.solve(Q_II, b)
    m[I] = m_I
    # m[B] = 0 by construction

    # Average MFPT over A (unweighted). You can also weight by equilibrium π if you prefer.
    mfpt = float(np.mean(m[A]))
    return mfpt


def leading_relaxation_times(Q: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute the k slowest relaxation times from the generator Q.

    Assumes Q is row-generator with one eigenvalue 0 and others negative.
    Returns an array of length <= k with t_n = -1 / lambda_n for nonzero eigenvalues.
    """
    Q = np.asarray(Q, dtype=float)
    # Full dense eigendecomposition; for big Q_eff, this is fine (M ~ O(10^2–10^3))
    evals, _ = np.linalg.eig(Q)
    # We want eigenvalues with largest real part (closest to 0), excluding ~0
    evals = np.real(evals)
    # Sort descending by real part
    evals_sorted = np.sort(evals)[::-1]

    # Filter out the zero eigenvalue (within tolerance)
    tol = 1e-10
    nonzero = evals_sorted[np.abs(evals_sorted) > tol]

    # Take up to k slowest modes (those closest to 0)
    slow = nonzero[:k]
    t = -1.0 / slow
    return t
