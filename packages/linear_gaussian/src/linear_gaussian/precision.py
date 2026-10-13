"""Precision-space numerical calculations for linear-Gaussian inference."""

import numpy as np

from .models import GaussianComponent, NDArrayFloat


def _block_diag(matrices: list[NDArrayFloat]) -> NDArrayFloat:
    """Build a block-diagonal matrix from a list of square matrices."""
    sizes = [m.shape[0] for m in matrices]
    n = sum(sizes)
    result = np.zeros((n, n))
    offset = 0
    for m in matrices:
        s = m.shape[0]
        result[offset : offset + s, offset : offset + s] = m
        offset += s
    return result


def _calc_mu_eta(nuisance: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Compute the nuisance mean mu_eta = sum_m A_m mu_m."""
    if not nuisance:
        return np.zeros(M)
    return np.sum([c.A @ c.mu for c in nuisance], axis=0)


def _calc_C_eta(nuisance: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Compute the nuisance covariance C_eta = sum_m A_m C_m A_m^T."""
    if not nuisance:
        return np.zeros((M, M))
    return np.sum([c.A @ c.C @ c.A.T for c in nuisance], axis=0)


def _build_A_I(inferred: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Build A_I = (A_1 | ... | A_k) by horizontal concatenation."""
    if not inferred:
        return np.zeros((M, 0))
    return np.hstack([c.A for c in inferred])


def _build_mu_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build mu_I = (mu_1; ...; mu_k) by vertical stacking."""
    if not inferred:
        return np.zeros(0)
    return np.concatenate([c.mu for c in inferred])


def _build_C_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build C_I = diag(C_1, ..., C_k) as a block-diagonal matrix."""
    if not inferred:
        return np.zeros((0, 0))
    return _block_diag([c.C for c in inferred])


def _solve_cholesky(L: NDArrayFloat, rhs: NDArrayFloat) -> NDArrayFloat:
    y = np.linalg.solve(L, rhs)
    return np.linalg.solve(L.T, y)


def _solve_symmetric_system(
    L: NDArrayFloat | None, matrix: NDArrayFloat, rhs: NDArrayFloat
) -> NDArrayFloat:
    if L is not None:
        return _solve_cholesky(L, rhs)

    try:
        L_chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Matrix is singular or not numerically positive definite."
            ) from exc
    else:
        return _solve_cholesky(L_chol, rhs)


def _calc_Lambda(
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    C_eta: NDArrayFloat,
    L_C_I: NDArrayFloat | None = None,
    L_C_eta: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Compute the posterior precision Lambda = C_I^{-1} + A_I^T C_eta^{-1} A_I."""
    identity = np.eye(C_I.shape[0], dtype=C_I.dtype)
    C_I_inv = _solve_symmetric_system(L_C_I, C_I, identity)
    C_eta_inv_A_I = _solve_symmetric_system(L_C_eta, C_eta, A_I)
    return C_I_inv + A_I.T @ C_eta_inv_A_I


def _calc_h(
    d: NDArrayFloat,
    mu_I: NDArrayFloat,
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    mu_eta: NDArrayFloat,
    C_eta: NDArrayFloat,
    L_C_I: NDArrayFloat | None = None,
    L_C_eta: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Compute the information vector h = C_I^{-1} mu_I + A_I^T C_eta^{-1} (d - mu_eta).

    If Cholesky factors are provided they will be used for the solves.
    """
    C_I_inv_mu_I = _solve_symmetric_system(L_C_I, C_I, mu_I)
    C_eta_inv_residual = _solve_symmetric_system(L_C_eta, C_eta, d - mu_eta)
    return C_I_inv_mu_I + A_I.T @ C_eta_inv_residual


def _log_gaussian_density(
    x: NDArrayFloat,
    mean: NDArrayFloat,
    cov: NDArrayFloat,
) -> float:
    """Evaluate the log of N(x; mean, cov)."""
    M = x.shape[0]
    diff = x - mean

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter_scale = float(np.trace(cov) / M)
        if not np.isfinite(jitter_scale) or jitter_scale <= 0:
            jitter_scale = 1.0
        jitter = max(1e-12 * jitter_scale, 1e-12)

        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(M))
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0 or not np.isfinite(log_det):
                raise ValueError(
                    "Covariance matrix is not positive definite."
                ) from None
            quad = float(diff @ np.linalg.solve(cov, diff))
            return float(-0.5 * M * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad)

    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    y = np.linalg.solve(L, diff)
    quad = float(y @ y)
    return float(-0.5 * M * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad)
