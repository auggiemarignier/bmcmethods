"""Public functions for linear-Gaussian inference."""

from . import data_space, precision
from .common import log_evidence
from .models import GaussianComponent, NDArrayFloat

_STRATEGIES = {"precision": precision, "data": data_space}


def _select_strategy(formulation: str):
    try:
        return _STRATEGIES[formulation]
    except (KeyError, TypeError):
        raise ValueError("formulation must be either 'precision' or 'data'.") from None


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
    *,
    formulation: str = "precision",
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
    strategy = _select_strategy(formulation)
    if not inferred:
        raise ValueError(
            "inferred must be non-empty to compute a posterior covariance."
        )
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior covariance; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )

    return strategy.posterior_cov(inferred, nuisance)


def calc_posterior_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    *,
    formulation: str = "precision",
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
    strategy = _select_strategy(formulation)
    if not inferred:
        raise ValueError("inferred must be non-empty to compute a posterior mean.")
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior mean; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )
    _validate_inputs(d, inferred, nuisance)

    return strategy.posterior_mean(d, inferred, nuisance)


def calc_log_evidence(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    *,
    formulation: str = "precision",
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
    return log_evidence(d, inferred, nuisance)


def calc_posterior_predictive_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    *,
    formulation: str = "precision",
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
    strategy = _select_strategy(formulation)
    if not inferred:
        raise ValueError("inferred must be non-empty to compute a posterior mean.")
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior mean; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )
    _validate_inputs(d, inferred, nuisance)
    return strategy.posterior_predictive_mean(d, inferred, nuisance)


def calc_posterior_predictive_cov(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    *,
    formulation: str = "precision",
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
    strategy = _select_strategy(formulation)
    if not inferred:
        raise ValueError(
            "inferred must be non-empty to compute a posterior covariance."
        )
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior covariance; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )
    return strategy.posterior_predictive_cov(inferred, nuisance)
