"""Small numerical helpers shared by the KTN checks."""

from __future__ import annotations

from collections.abc import Iterable
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import ArpackNoConvergence, MatrixRankWarning, eigs, spsolve


def _as_square_csr(Q: np.ndarray | sp.spmatrix) -> sp.csr_matrix:
    matrix = sp.csr_matrix(Q, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if matrix.data.size and not np.all(np.isfinite(matrix.data)):
        raise ValueError("Q contains non-finite values.")
    return matrix


def _state_indices(states: Iterable[int], n: int, name: str) -> np.ndarray:
    indices = np.array(sorted({int(i) for i in states}), dtype=int)
    if indices.size == 0:
        raise ValueError(f"{name} must contain at least one state.")
    if indices[0] < 0 or indices[-1] >= n:
        raise IndexError(f"{name} contains a state outside 0..{n - 1}.")
    return indices


def _check_generator(Q: sp.csr_matrix, column_generator: bool) -> None:
    axis = 0 if column_generator else 1
    sums = np.asarray(Q.sum(axis=axis)).ravel()
    scale = max(float(np.max(np.abs(Q.data))) if Q.nnz else 0.0, 1.0)
    if sums.size and float(np.max(np.abs(sums))) > 1e-8 * scale:
        orientation = "columns" if column_generator else "rows"
        raise ValueError(f"Q {orientation} do not sum to zero within tolerance.")


def compute_mfpt_from_Q(
    Q: np.ndarray | sp.spmatrix,
    A: Iterable[int],
    B: Iterable[int],
    *,
    column_generator: bool = True,
) -> float:
    """Return the uniformly averaged MFPT from A to B.

    Project matrices use zero column sums. Set ``column_generator=False`` for
    the usual row-generator form.
    """
    matrix = _as_square_csr(Q)
    _check_generator(matrix, column_generator)

    n = matrix.shape[0]
    A_idx = _state_indices(A, n, "A")
    B_idx = _state_indices(B, n, "B")

    backward = matrix.T.tocsr() if column_generator else matrix
    interior_mask = np.ones(n, dtype=bool)
    interior_mask[B_idx] = False
    interior = np.flatnonzero(interior_mask)

    hitting_times = np.zeros(n, dtype=float)
    if interior.size:
        system = backward[interior][:, interior].tocsc()
        rhs = -np.ones(interior.size, dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                solution = np.asarray(spsolve(system, rhs), dtype=float)
            except (MatrixRankWarning, RuntimeError) as exc:
                raise np.linalg.LinAlgError(
                    "MFPT system is singular. B may be unreachable from some states."
                ) from exc

        if not np.all(np.isfinite(solution)):
            raise np.linalg.LinAlgError(
                "MFPT solve returned non-finite values. B may be unreachable."
            )

        negative_tol = 1e-9 * max(float(np.max(np.abs(solution))), 1.0)
        if np.any(solution < -negative_tol):
            raise np.linalg.LinAlgError("MFPT solve returned negative passage times.")
        hitting_times[interior] = np.clip(solution, 0.0, None)

    return float(np.mean(hitting_times[A_idx]))


def leading_relaxation_times(
    Q: np.ndarray | sp.spmatrix,
    k: int = 5,
    *,
    zero_tol: float = 1e-12,
) -> np.ndarray:
    """Return the slowest positive relaxation times ``-1 / Re(lambda)``."""
    matrix = _as_square_csr(Q)
    if k < 0:
        raise ValueError("k must be non-negative.")
    if k == 0 or matrix.shape[0] <= 1:
        return np.array([], dtype=float)

    n = matrix.shape[0]
    if n <= 64 or k >= n - 1:
        values = np.linalg.eigvals(matrix.toarray())
    else:
        n_values = min(max(k + 2, 3), n - 2)
        try:
            values = eigs(
                matrix,
                k=n_values,
                which="LR",
                return_eigenvectors=False,
                tol=1e-9,
                maxiter=max(10000, 100 * n),
            )
        except ArpackNoConvergence as exc:
            values = exc.eigenvalues
            if values is None or len(values) < min(k + 1, n_values):
                values = eigs(
                    matrix,
                    k=n_values,
                    sigma=-1e-12,
                    which="LM",
                    return_eigenvectors=False,
                    tol=1e-9,
                    maxiter=max(20000, 200 * n),
                )

    real_parts = np.real(values)
    slow = np.sort(real_parts[real_parts < -abs(zero_tol)])[::-1][:k]
    return -1.0 / slow
