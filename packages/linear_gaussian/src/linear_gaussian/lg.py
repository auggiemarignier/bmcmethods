"""Bayesian inference for linear combinations of Gaussian random variables.

Implements posterior mean, posterior covariance, and log-evidence for the
linear-Gaussian model

    d = A_I X_I + eta

where X_I is the inferred variable, A_I is the forward operator, and eta is
the aggregate nuisance contribution.  Both X_I and eta are Gaussian.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type NDArrayFloat = NDArray[np.float64]


@dataclass
class GaussianComponent:
    """A single Gaussian component in the linear-Gaussian model.

    Parameters
    ----------
    A : ndarray, shape (M, N_j)
        Linear mapping from the N_j-dimensional latent variable to the
        M-dimensional observation space.
    mu : ndarray, shape (N_j,)
        Prior mean of the latent variable.
    C : ndarray, shape (N_j, N_j)
        Prior covariance of the latent variable.
    """

    A: NDArrayFloat
    mu: NDArrayFloat
    C: NDArrayFloat

    def __post_init__(self) -> None:
        if self.A.ndim != 2:
            raise ValueError("A must be a 2D array")
        if self.mu.ndim != 1:
            raise ValueError("mu must be a 1D array")
        if self.C.ndim != 2 or self.C.shape[0] != self.C.shape[1]:
            raise ValueError("C must be a 2D square matrix")
        if self.mu.shape[0] != self.C.shape[0]:
            raise ValueError("mu and C dimensions are incompatible")
        if self.A.shape[1] != self.mu.shape[0]:
            raise ValueError("A columns and mu are incompatible")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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


def _build_A_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build A_I = (A_1 | ... | A_k) by horizontal concatenation."""
    return np.hstack([c.A for c in inferred])


def _build_mu_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build mu_I = (mu_1; ...; mu_k) by vertical stacking."""
    return np.concatenate([c.mu for c in inferred])


def _build_C_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build C_I = diag(C_1, ..., C_k) as a block-diagonal matrix."""
    return _block_diag([c.C for c in inferred])


def _calc_Lambda(
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    C_eta: NDArrayFloat,
) -> NDArrayFloat:
    """Compute the posterior precision Lambda = C_I^{-1} + A_I^T C_eta^{-1} A_I."""
    C_I_inv = np.linalg.inv(C_I)
    C_eta_inv = np.linalg.inv(C_eta)
    return C_I_inv + A_I.T @ C_eta_inv @ A_I


def _calc_h(
    d: NDArrayFloat,
    mu_I: NDArrayFloat,
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    mu_eta: NDArrayFloat,
    C_eta: NDArrayFloat,
) -> NDArrayFloat:
    """Compute the information vector h = C_I^{-1} mu_I + A_I^T C_eta^{-1} (d - mu_eta)."""
    C_I_inv = np.linalg.inv(C_I)
    C_eta_inv = np.linalg.inv(C_eta)
    return C_I_inv @ mu_I + A_I.T @ C_eta_inv @ (d - mu_eta)


def _infer_M(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> int:
    """Infer the observation-space dimension M from the components."""
    all_components = inferred + nuisance
    return all_components[0].A.shape[0]


def _validate_inputs(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> None:
    """Validate that d, inferred and nuisance are mutually consistent."""
    if not inferred and not nuisance:
        raise ValueError(
            "At least one of inferred or nuisance must be non-empty."
        )
    M = _infer_M(inferred, nuisance)
    for comp in inferred + nuisance:
        if comp.A.shape[0] != M:
            raise ValueError(
                f"All components must map to the same observation dimension M={M}, "
                f"but found A with shape {comp.A.shape}."
            )
    if d.shape[0] != M:
        raise ValueError(
            f"d has dimension {d.shape[0]} but components map to dimension M={M}."
        )


def _log_gaussian_density(
    x: NDArrayFloat,
    mean: NDArrayFloat,
    cov: NDArrayFloat,
) -> float:
    """Evaluate the log of N(x; mean, cov)."""
    M = x.shape[0]
    diff = x - mean
    sign, log_det = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite.")
    return float(
        -0.5 * M * np.log(2 * np.pi)
        - 0.5 * log_det
        - 0.5 * diff @ np.linalg.inv(cov) @ diff
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calc_posterior_cov(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Compute the posterior covariance of the inferred variables.

    Parameters
    ----------
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    C_post : ndarray, shape (N_I, N_I)
        Posterior covariance, where N_I = sum_j N_j over inferred components.

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    if not inferred:
        raise ValueError(
            "inferred must be non-empty to compute a posterior covariance."
        )
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior covariance; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )

    A_I = _build_A_I(inferred)
    C_I = _build_C_I(inferred)
    M = A_I.shape[0]
    mu_eta = _calc_mu_eta(nuisance, M)  # noqa: F841 – not needed here but kept for clarity
    C_eta = _calc_C_eta(nuisance, M)

    Lambda = _calc_Lambda(C_I, A_I, C_eta)
    return np.linalg.inv(Lambda)


def calc_posterior_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Compute the posterior mean of the inferred variables.

    Parameters
    ----------
    d : ndarray, shape (M,)
        Observed data vector.
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    mu_post : ndarray, shape (N_I,)
        Posterior mean, where N_I = sum_j N_j over inferred components.

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    if not inferred:
        raise ValueError(
            "inferred must be non-empty to compute a posterior mean."
        )
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior mean; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )

    A_I = _build_A_I(inferred)
    mu_I = _build_mu_I(inferred)
    C_I = _build_C_I(inferred)
    M = A_I.shape[0]
    mu_eta = _calc_mu_eta(nuisance, M)
    C_eta = _calc_C_eta(nuisance, M)

    Lambda = _calc_Lambda(C_I, A_I, C_eta)
    h = _calc_h(d, mu_I, C_I, A_I, mu_eta, C_eta)
    C_post = np.linalg.inv(Lambda)
    return C_post @ h


def calc_log_evidence(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> float:
    """Compute the log marginal likelihood (log evidence) log p(d).

    The evidence is the marginal likelihood obtained by integrating out all
    latent variables.  Because the sum of linear transformations of independent
    Gaussians is itself Gaussian, the marginal distribution of the data is

        d ~ N(A_I mu_I + mu_eta,  A_I C_I A_I^T + C_eta)

    where nuisance variables have been analytically marginalised to yield
    mu_eta and C_eta.

    Parameters
    ----------
    d : ndarray, shape (M,)
        Observed data vector.
    inferred : list of GaussianComponent
        The inferred components.  May be empty, in which case the evidence
        reduces to the nuisance marginal N(d; mu_eta, C_eta).
    nuisance : list of GaussianComponent
        The nuisance components.  May be empty, in which case the evidence
        uses only the inferred prior marginal N(d; A_I mu_I, A_I C_I A_I^T).

    Returns
    -------
    log_Z : float
        Log marginal likelihood log p(d).

    Raises
    ------
    ValueError
        If both ``inferred`` and ``nuisance`` are empty.
    """
    _validate_inputs(d, inferred, nuisance)

    M = _infer_M(inferred, nuisance)

    if not inferred:
        # Only nuisance: d ~ N(mu_eta, C_eta)
        mu_marginal = _calc_mu_eta(nuisance, M)
        C_marginal = _calc_C_eta(nuisance, M)
    elif not nuisance:
        # Only inferred, no noise: d ~ N(A_I mu_I, A_I C_I A_I^T)
        A_I = _build_A_I(inferred)
        mu_I = _build_mu_I(inferred)
        C_I = _build_C_I(inferred)
        mu_marginal = A_I @ mu_I
        C_marginal = A_I @ C_I @ A_I.T
    else:
        # General case
        A_I = _build_A_I(inferred)
        mu_I = _build_mu_I(inferred)
        C_I = _build_C_I(inferred)
        mu_eta = _calc_mu_eta(nuisance, M)
        C_eta = _calc_C_eta(nuisance, M)
        mu_marginal = A_I @ mu_I + mu_eta
        C_marginal = A_I @ C_I @ A_I.T + C_eta

    return _log_gaussian_density(d, mu_marginal, C_marginal)
