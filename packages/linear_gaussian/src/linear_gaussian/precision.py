"""Precision-space numerical calculations for linear-Gaussian inference."""

import numpy as np

from .cache import get_precision, store_precision
from .common import posterior_predictive_cov as ppc
from .common import posterior_predictive_mean as ppm
from .common import prepare, solve_symmetric
from .models import GaussianComponent, NDArrayFloat


def _cholesky_or_none(matrix: NDArrayFloat) -> NDArrayFloat | None:
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return None


def _calc_Lambda(
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    C_eta: NDArrayFloat,
    L_C_I: NDArrayFloat | None = None,
    L_C_eta: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Compute the posterior precision Lambda = C_I^{-1} + A_I^T C_eta^{-1} A_I."""
    identity = np.eye(C_I.shape[0], dtype=C_I.dtype)
    C_I_inv = solve_symmetric(C_I, identity, L_C_I)
    C_eta_inv_A_I = solve_symmetric(C_eta, A_I, L_C_eta)
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
    C_I_inv_mu_I = solve_symmetric(C_I, mu_I, L_C_I)
    C_eta_inv_residual = solve_symmetric(C_eta, d - mu_eta, L_C_eta)
    return C_I_inv_mu_I + A_I.T @ C_eta_inv_residual


def _precision_and_factor(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> tuple[
    NDArrayFloat | None,
    NDArrayFloat | None,
    NDArrayFloat,
    NDArrayFloat,
]:
    """Return cached prior, nuisance, and posterior precision factors."""
    cached = get_precision(inferred, nuisance)
    if cached is not None:
        return cached

    A_I, _, C_I, _, C_eta = prepare(inferred, nuisance)
    L_C_I = _cholesky_or_none(C_I)
    L_C_eta = _cholesky_or_none(C_eta)
    Lambda = _calc_Lambda(C_I, A_I, C_eta, L_C_I, L_C_eta)
    L_Lambda = np.linalg.cholesky(Lambda)
    result = (L_C_I, L_C_eta, Lambda, L_Lambda)
    store_precision(inferred, nuisance, result)
    return result


def posterior_cov(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> NDArrayFloat:
    """Posterior covariance computed with the latent precision matrix."""
    _, _, Lambda, L_Lambda = _precision_and_factor(inferred, nuisance)
    return solve_symmetric(Lambda, np.eye(Lambda.shape[0]), L_Lambda)


def posterior_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Posterior mean computed with the latent precision matrix."""
    A_I, mu_I, C_I, mu_eta, C_eta = prepare(inferred, nuisance)
    L_C_I, L_C_eta, Lambda, L_Lambda = _precision_and_factor(inferred, nuisance)
    h = _calc_h(d, mu_I, C_I, A_I, mu_eta, C_eta, L_C_I, L_C_eta)
    return solve_symmetric(Lambda, h, L_Lambda)


def posterior_predictive_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Posterior predictive mean using the precision-space posterior."""
    return ppm(d, inferred, nuisance, posterior_mean)


def posterior_predictive_cov(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> NDArrayFloat:
    """Posterior predictive covariance using the precision-space posterior."""
    return ppc(inferred, nuisance, posterior_cov)
