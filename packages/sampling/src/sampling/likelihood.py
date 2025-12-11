"""Likelihood functions of MCMC sampling."""

from collections.abc import Callable

import numpy as np


class GaussianLikelihood:
    """
    Represents a Gaussian likelihood function for MCMC sampling.

    The Gaussian likelihood assumes that the observed data is normally distributed
    around the model predictions, with a specified inverse covariance matrix.
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        observed_data: np.ndarray,
        inv_covar: np.ndarray,
        validate_covariance: bool = True,
        example_model: None | np.ndarray = None,
    ) -> None:
        """
        Initialize the Gaussian likelihood.

        Parameters
        ----------
        forward_fn : Callable[[np.ndarray], np.ndarray]
            Forward model function that takes model parameters and returns predicted data.
        observed_data : ndarray, shape (n,)
            Observed data.
        inv_covar : ndarray, shape (n, n)
            Inverse covariance matrix of the observed data. Must be symmetric and positive semidefinite.
        validate_covariance : bool, optional
            Whether to validate the inverse covariance matrix. Default is True.
        example_model : None | ndarray, optional
            Example model parameters to validate the forward function. If None (default), no validation is performed.

        Raises
        ------
        ValueError
            If the inverse covariance matrix is not symmetric or not positive semidefinite.
        """
        _validate_data_vector(observed_data)
        if validate_covariance:
            _validate_covariance_matrix(inv_covar, observed_data.size)
        if example_model is not None:
            _validate_forward_function(forward_fn, example_model, observed_data.size)

        self.forward_fn = forward_fn
        self.observed_data = observed_data
        self.inv_covar = inv_covar

    def __call__(self, model_params: np.ndarray) -> float:
        """
        Evaluate the log-likelihood for given model parameters.

        Parameters
        ----------
        model_params : ndarray
            Model parameters.

        Returns
        -------
        log_likelihood : float
            The log-likelihood value.
        """
        predicted_data = self.forward_fn(model_params)
        residual = self.observed_data - predicted_data
        return -0.5 * residual.T @ self.inv_covar @ residual


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
    covar : ndarray, shape (n, n)
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
    predicted_data = forward_fn(example_model)
    if predicted_data.shape != (N,):
        raise ValueError(f"Forward function must return prediction of shape ({N},).")
