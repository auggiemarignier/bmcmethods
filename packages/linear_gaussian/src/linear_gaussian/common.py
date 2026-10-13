"""Shared assembly and Gaussian linear algebra for inference strategies."""

from collections.abc import Callable

import numpy as np

from .models import GaussianComponent, NDArrayFloat


def _block_diag(matrices: list[NDArrayFloat]) -> NDArrayFloat:
    """Build a block-diagonal matrix from square matrices."""
    sizes = [matrix.shape[0] for matrix in matrices]
    result = np.zeros((sum(sizes), sum(sizes)))
    offset = 0
    for matrix, size in zip(matrices, sizes, strict=True):
        result[offset : offset + size, offset : offset + size] = matrix
        offset += size
    return result


def build_A_I(inferred: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Build the joint inferred forward operator by horizontal concatenation."""
    if not inferred:
        return np.zeros((M, 0))
    return np.hstack([component.A for component in inferred])


def build_mu_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build the joint inferred prior mean by vertical concatenation."""
    if not inferred:
        return np.zeros(0)
    return np.concatenate([component.mu for component in inferred])


def build_C_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build the joint inferred prior covariance as a block diagonal matrix."""
    if not inferred:
        return np.zeros((0, 0))
    return _block_diag([component.C for component in inferred])


def calc_mu_eta(nuisance: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Compute the aggregate nuisance mean."""
    if not nuisance:
        return np.zeros(M)
    return np.sum([component.A @ component.mu for component in nuisance], axis=0)


def calc_C_eta(nuisance: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Compute the aggregate nuisance covariance."""
    if not nuisance:
        return np.zeros((M, M))
    return np.sum(
        [component.A @ component.C @ component.A.T for component in nuisance],
        axis=0,
    )


def prepare(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Assemble shared arrays, reusing a cached result when available."""
    from .cache import get_prepared, store_prepared

    prepared = get_prepared(inferred, nuisance)
    if prepared is not None:
        return prepared

    M = (inferred + nuisance)[0].A.shape[0]
    prepared = (
        build_A_I(inferred, M),
        build_mu_I(inferred),
        build_C_I(inferred),
        calc_mu_eta(nuisance, M),
        calc_C_eta(nuisance, M),
    )
    store_prepared(inferred, nuisance, prepared)
    return prepared


def solve_symmetric(
    matrix: NDArrayFloat,
    rhs: NDArrayFloat,
    factor: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Solve a symmetric positive definite system, with a dense fallback."""
    if factor is not None:
        return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    try:
        factor = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Matrix is singular or not numerically positive definite."
            ) from exc
    return np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))


def log_gaussian_density(
    x: NDArrayFloat, mean: NDArrayFloat, covariance: NDArrayFloat
) -> float:
    """Evaluate a multivariate Gaussian log density."""
    dimension = x.shape[0]
    residual = x - mean
    try:
        factor = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError:
        jitter_scale = float(np.trace(covariance) / dimension)
        if not np.isfinite(jitter_scale) or jitter_scale <= 0:
            jitter_scale = 1.0
        jitter = max(1e-12 * jitter_scale, 1e-12)
        try:
            factor = np.linalg.cholesky(covariance + jitter * np.eye(dimension))
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(covariance)
            if sign <= 0 or not np.isfinite(log_det):
                raise ValueError(
                    "Covariance matrix is not positive definite."
                ) from None
            quadratic = float(residual @ np.linalg.solve(covariance, residual))
            return float(
                -0.5 * dimension * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quadratic
            )
    log_det = 2.0 * np.sum(np.log(np.diag(factor)))
    whitened = np.linalg.solve(factor, residual)
    return float(
        -0.5 * dimension * np.log(2 * np.pi)
        - 0.5 * log_det
        - 0.5 * (whitened @ whitened)
    )


def log_evidence(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> float:
    """Compute evidence from the marginal Gaussian over observations."""
    A_I, mu_I, C_I, mu_eta, C_eta = prepare(inferred, nuisance)
    mean = A_I @ mu_I + mu_eta
    covariance = A_I @ C_I @ A_I.T + C_eta
    return log_gaussian_density(d, mean, covariance)


def posterior_predictive_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    mean_fn: Callable[
        [NDArrayFloat, list[GaussianComponent], list[GaussianComponent]], NDArrayFloat
    ],
) -> NDArrayFloat:
    """Posterior predictive mean using the data-space posterior."""
    A_I, _, _, mu_eta, _ = prepare(inferred, nuisance)
    return A_I @ mean_fn(d, inferred, nuisance) + mu_eta


def posterior_predictive_cov(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    cov_fn: Callable[[list[GaussianComponent], list[GaussianComponent]], NDArrayFloat],
) -> NDArrayFloat:
    """Posterior predictive covariance using the data-space posterior."""
    A_I, _, _, _, C_eta = prepare(inferred, nuisance)
    return A_I @ cov_fn(inferred, nuisance) @ A_I.T + C_eta
