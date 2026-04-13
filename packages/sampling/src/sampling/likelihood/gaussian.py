"""Likelihood functions of MCMC sampling."""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Self

import numpy as np

from ._base import ForwardFunction, ForwardGradientFunction, LikelihoodBase


class CovarianceKind(StrEnum):
    """Enumeration for the type of covariance matrix used in the Gaussian likelihood."""

    FULL = auto()
    DIAG = auto()
    SCALAR = auto()


@dataclass(frozen=True)
class GaussianLikelihoodState:
    """Container for the state of the Gaussian likelihood, used for pickling when using multiprocessing."""

    observed_data: np.ndarray
    inv_covar: np.ndarray
    covar_kind: CovarianceKind = field(init=False)

    def __post_init__(self) -> None:
        """Determine the kind of covariance matrix based on the shape of the inverse covariance."""
        if self.inv_covar.ndim == 2:
            kind = CovarianceKind.FULL
        elif self.inv_covar.ndim == 1 and self.inv_covar.size > 1:
            kind = CovarianceKind.DIAG
        else:
            kind = CovarianceKind.SCALAR
        object.__setattr__(self, "covar_kind", kind)


def gaussian_log_likelihood(
    model_params: np.ndarray,
    forward_fn: ForwardFunction,
    state: GaussianLikelihoodState,
) -> np.ndarray:
    """Compute the Gaussian log-likelihood for given model parameters."""
    model_params = np.atleast_2d(model_params)
    predicted = forward_fn(model_params)
    residuals = state.observed_data[None, :] - predicted

    if state.covar_kind == CovarianceKind.FULL:
        # Doing it this way avoids computing the full (batch x batch) matrix of residuals @ inv_covar @ residuals.T and then discarding all the off-diagonal terms
        out = -0.5 * np.sum((residuals @ state.inv_covar) * residuals, axis=1)
    elif state.covar_kind == CovarianceKind.DIAG:
        out = -0.5 * np.sum(residuals**2 * state.inv_covar[None, :], axis=1)
    else:
        out = -0.5 * np.sum(residuals**2, axis=1) * state.inv_covar[0]

    return out.squeeze()


def grad_gaussian_loglikelihood(
    model_params: np.ndarray,
    forward_fn: ForwardFunction,
    forward_fn_gradient: ForwardGradientFunction,
    state: GaussianLikelihoodState,
) -> np.ndarray:
    """Compute the gradient of the Gaussian log-likelihood with respect to model parameters."""
    model_params = np.atleast_2d(model_params)
    predicted = forward_fn(model_params)
    residuals = state.observed_data[None, :] - predicted

    if state.covar_kind == CovarianceKind.FULL:
        weighted_residuals = np.einsum("bi,ij->bj", residuals, state.inv_covar)
    elif state.covar_kind == CovarianceKind.DIAG:
        weighted_residuals = residuals * state.inv_covar
    else:
        weighted_residuals = residuals * state.inv_covar.item()

    J = forward_fn_gradient(model_params)
    if J.ndim == 2:
        J = J[None, :, :]

    gradient = np.einsum("bni,bn->bi", J, weighted_residuals)
    return gradient.squeeze()


class GaussianLikelihood(LikelihoodBase[GaussianLikelihoodState]):
    """
    Represents a Gaussian likelihood function for MCMC sampling.

    The Gaussian likelihood assumes that the observed data is normally distributed around the model predictions, with a specified inverse covariance matrix.
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        observed_data: np.ndarray,
        inv_covar: float | np.ndarray,
        forward_fn_gradient: None | Callable[[np.ndarray], np.ndarray] = None,
        validate_covariance: bool = True,
        example_model: None | np.ndarray = None,
    ) -> None:
        """
        Initialise the Gaussian likelihood.

        Parameters
        ----------
        forward_fn : Callable[[np.ndarray], np.ndarray]
            Forward model function that takes model parameters and returns predicted data.
            Should accept shape (..., ndim) and return shape (..., n).
        observed_data : ndarray, shape (n,)
            Observed data.
        inv_covar : float | ndarray, shape (1,), (n,) or (n, n)
            Inverse covariance matrix of the observed data. Either a full matrix of shape (n, n),
            a diagonal represented as a vector of shape (n,), or a scalar representing uniform variance.
        validate_covariance : bool, optional
            Whether to validate the inverse covariance matrix. Default is True.
        example_model : None | ndarray, optional
            Example model parameters to validate the forward function. If None (default), no validation is performed.
        forward_fn_gradient : None | Callable[[np.ndarray], np.ndarray], optional
            Gradient of the forward function with respect to model parameters. If None (default), no gradient is available.

        Raises
        ------
        ValueError
            If the inverse covariance matrix is not symmetric or not positive semidefinite.
        """
        _validate_data_vector(observed_data)
        inv_covar = np.array(inv_covar)
        if validate_covariance:
            _validate_covariance_matrix(inv_covar, observed_data.size)
        if example_model is not None:
            _validate_forward_function(forward_fn, example_model, observed_data.size)

        self.forward_fn = forward_fn
        self.forward_fn_gradient = forward_fn_gradient

        self.state = GaussianLikelihoodState(
            observed_data=observed_data,
            inv_covar=inv_covar,
        )

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """
        Evaluate the log-likelihood for given model parameters.

        Parameters
        ----------
        model_params : ndarray, shape (ndim,) or (batch, ndim)
            Model parameters. Can be a single parameter set or a batch.

        Returns
        -------
        log_likelihood : ndarray
            The log-likelihood value(s).
        """
        return gaussian_log_likelihood(model_params, self.forward_fn, self.state)

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the log-likelihood with respect to model parameters.

        Parameters
        ----------
        model_params : ndarray, shape (ndim,) or (batch, ndim)
            Model parameters. Can be a single parameter set or a batch.

        Returns
        -------
        gradient : ndarray
            The gradient of the log-likelihood with respect to model parameters.
        """
        if self.forward_fn_gradient is None:
            raise RuntimeError(
                "Gradient function for the forward model must be provided to compute the likelihood gradient."
            )
        return grad_gaussian_loglikelihood(
            model_params, self.forward_fn, self.forward_fn_gradient, self.state
        )

    @classmethod
    def from_state(
        cls,
        state: GaussianLikelihoodState,
        *,
        forward_fn: ForwardFunction | None = None,
        forward_fn_gradient: ForwardGradientFunction | None = None,
    ) -> Self:
        """Initialise from a state object.

        Useful for initialising in multiple workers.
        """
        if forward_fn is None:
            raise ValueError("Forward model required")
        return cls(
            forward_fn,
            state.observed_data,
            state.inv_covar,
            forward_fn_gradient,
        )


def _validate_data_vector(data: np.ndarray) -> None:
    """
    Validate that the data vector is one-dimensional.

    Parameters
    ----------
    data : ndarray
        Data vector to validate.

    Raises
    ------
    ValueError
        If the data vector is not one-dimensional.
    """
    if data.ndim != 1:
        raise ValueError("Data vector must be one-dimensional.")


def _validate_covariance_matrix(covar: np.ndarray, N: int) -> None:
    """
    Validate that the inverse covariance matrix is symmetric and positive semidefinite.

    Parameters
    ----------
    covar : ndarray, shape (1,), (n,) or (n, n)
        Inverse covariance matrix to validate.
    N : int
        Expected size of the inverse covariance matrix.

    Raises
    ------
    ValueError
        If the inverse covariance matrix
            - has incorrect shape;
            - is not symmetric; or
            - is not positive semidefinite.
    """
    if covar.shape == (1,):
        if covar[0] <= 0:
            raise ValueError("Variance scalar must be positive.")
        return

    if covar.shape == (N,):
        if np.any(covar <= 0):
            raise ValueError("Covariance diagonal elements must be positive.")
        return

    if covar.shape != (N, N):
        raise ValueError(f"Covariance matrix must be of shape ({N}, {N}).")

    if not np.allclose(covar, covar.T):
        raise ValueError("Covariance matrix must be symmetric.")

    try:
        np.linalg.cholesky(covar)
        # If Cholesky decomposition succeeds, the matrix is positive definite
        # It is very unlikely for a realistic inverse covariance matrix to have zero eigenvalues (positive semidefinite) so this check is sufficient
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Inverse covariance matrix must be positive semidefinite."
        ) from e


def _validate_forward_function(
    forward_fn: Callable[[np.ndarray], np.ndarray], example_model: np.ndarray, N: int
) -> None:
    """
    Validate that the forward function returns data of the correct shape.

    Parameters
    ----------
    forward_fn : Callable[[np.ndarray], np.ndarray]
        Forward model function to validate.
    example_model : ndarray
        Example model parameters to test the forward function.
    N : int
        Expected size of the output data vector.

    Raises
    ------
    ValueError
        If the forward function does not return data of the correct shape.
    """
    example_batch = example_model[None, :]
    predicted_data = forward_fn(example_batch)
    if predicted_data.shape != (1, N):
        raise ValueError(
            f"Forward function must return prediction of shape (batch, {N})."
        )
