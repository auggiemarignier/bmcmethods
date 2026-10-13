"""Public functions for linear-Gaussian inference."""

import numpy as np

from .cache import _prepare_problem
from .models import GaussianComponent, NDArrayFloat
from .precision import (
    _calc_h,
    _log_gaussian_density,
    _solve_symmetric_system,
)


def _validate_inputs(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> None:
    """Validate that d, inferred and nuisance are mutually consistent."""
    if not inferred and not nuisance:
        raise ValueError("At least one of inferred or nuisance must be non-empty.")
    M = (inferred + nuisance)[0].A.shape[0]
    for comp in inferred + nuisance:
        if comp.A.shape[0] != M:
            raise ValueError(
                f"All components must map to the same observation dimension M={M}, "
                f"but found A with shape {comp.A.shape}."
            )
    if d.ndim != 1:
        raise ValueError(f"d must be a 1D array, but has shape {d.shape}.")
    if d.shape[0] != M:
        raise ValueError(
            f"d has dimension {d.shape[0]} but components map to dimension M={M}."
        )


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

    prepared = _prepare_problem(inferred, nuisance)
    if prepared.Lambda is None:
        raise RuntimeError("_prepare_problem did not compute Lambda.")
    identity = np.eye(prepared.Lambda.shape[0], dtype=prepared.Lambda.dtype)
    return _solve_symmetric_system(prepared.L_Lambda, prepared.Lambda, identity)


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
        raise ValueError("inferred must be non-empty to compute a posterior mean.")
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior mean; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )
    _validate_inputs(d, inferred, nuisance)

    prepared = _prepare_problem(inferred, nuisance)
    if prepared.Lambda is None:
        raise RuntimeError("_prepare_problem did not compute Lambda.")
    h = _calc_h(
        d,
        prepared.mu_I,
        prepared.C_I,
        prepared.A_I,
        prepared.mu_eta,
        prepared.C_eta,
        prepared.L_C_I,
        prepared.L_C_eta,
    )
    return _solve_symmetric_system(prepared.L_Lambda, prepared.Lambda, h)


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
    prepared = _prepare_problem(inferred, nuisance)

    if not inferred:
        # Only nuisance: d ~ N(mu_eta, C_eta)
        mu_marginal = prepared.mu_eta
        C_marginal = prepared.C_eta
    elif not nuisance:
        # Only inferred, no noise: d ~ N(A_I mu_I, A_I C_I A_I^T)
        A_I = prepared.A_I
        mu_I = prepared.mu_I
        C_I = prepared.C_I
        mu_marginal = A_I @ mu_I
        C_marginal = A_I @ C_I @ A_I.T
    else:
        # General case
        A_I = prepared.A_I
        mu_I = prepared.mu_I
        C_I = prepared.C_I
        mu_eta = prepared.mu_eta
        C_eta = prepared.C_eta
        mu_marginal = A_I @ mu_I + mu_eta
        C_marginal = A_I @ C_I @ A_I.T + C_eta

    return _log_gaussian_density(d, mu_marginal, C_marginal)


def calc_posterior_predictive_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Compute the mean of the posterior predictive distribution.

    Computes the mean of the posterior predictive distribution d^* | d where d^* is a new replicate of the original data.

        mu_pred = A_I mu_post + mu_eta

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
    mu_pred : ndarray, shape (M,)
        Posterior predictive mean

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    mu_post = calc_posterior_mean(d, inferred, nuisance)
    prepared = _prepare_problem(inferred, nuisance)
    return prepared.A_I @ mu_post + prepared.mu_eta


def calc_posterior_predictive_cov(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> NDArrayFloat:
    """Compute the covariance of the posterior predictive distribution.

    Computes the covariance of the posterior predictive distribution d^* | d where d^* is a new replicate of the original data.

        C_pred = A_I C_post A_I^T + C_eta

    Parameters
    ----------
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    C_pred : ndarray, shape (M, M)
        Posterior predictive covariance of a new replicate d^*, where M is the
        original data length.

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    C_post = calc_posterior_cov(inferred, nuisance)
    prepared = _prepare_problem(inferred, nuisance)
    return prepared.A_I @ C_post @ prepared.A_I.T + prepared.C_eta
