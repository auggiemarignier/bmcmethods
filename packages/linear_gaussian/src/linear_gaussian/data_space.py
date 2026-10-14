"""Data-space strategy for linear-Gaussian inference."""

from .common import posterior_predictive_cov as ppc
from .common import posterior_predictive_mean as ppm
from .common import prepare, solve_symmetric
from .models import GaussianComponent, NDArrayFloat


def posterior_cov(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> NDArrayFloat:
    """Posterior covariance computed by conditioning in observation space."""
    A_I, _, C_I, _, C_eta = prepare(inferred, nuisance)
    data_cov = A_I @ C_I @ A_I.T + C_eta
    cross_cov = C_I @ A_I.T
    solved = solve_symmetric(data_cov, A_I @ C_I)
    return C_I - cross_cov @ solved


def posterior_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Posterior mean computed by conditioning in observation space."""
    A_I, mu_I, C_I, mu_eta, C_eta = prepare(inferred, nuisance)
    data_cov = A_I @ C_I @ A_I.T + C_eta
    residual = d - A_I @ mu_I - mu_eta
    cross_cov = C_I @ A_I.T
    return mu_I + cross_cov @ solve_symmetric(data_cov, residual)


def posterior_predictive_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Posterior predictive mean using the data-space posterior."""
    return ppm(d, inferred, nuisance, posterior_mean)


def posterior_predictive_cov(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> NDArrayFloat:
    """Posterior predictive covariance using the data-space posterior."""
    return ppc(inferred, nuisance, posterior_cov)
