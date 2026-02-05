"""Likelihood functions of MCMC sampling."""

from collections.abc import Callable

import numpy as np


class GaussianLikelihood:
    """
    Represents a Gaussian likelihood function for MCMC sampling.

    The Gaussian likelihood assumes that the observed data is normally distributed around the model predictions, with a specified inverse covariance matrix.
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        observed_data: np.ndarray,
        inv_covar: float | np.ndarray,
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

        Raises
        ------
        ValueError
            If the inverse covariance matrix is not symmetric or not positive semidefinite.
        """
        _validate_data_vector(observed_data)
        if validate_covariance:
            _validate_covariance_matrix(np.array(inv_covar), observed_data.size)
        if example_model is not None:
            _validate_forward_function(forward_fn, example_model, observed_data.size)

        self.forward_fn = forward_fn
        self.observed_data = observed_data
        self.inv_covar = np.array(inv_covar)
        self._exp_term_fn = self._choose_exponential_term_function()

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
        model_params = np.atleast_2d(model_params)  # Ensure shape is (batch, ndim)
        predicted = self.forward_fn(model_params)
        residuals = self.observed_data[None, :] - predicted
        return self._exp_term_fn(residuals).squeeze()  # Return scalar if input was 1D

    def _choose_exponential_term_function(self) -> Callable[[np.ndarray], np.ndarray]:
        """
        Choose the appropriate exponential term computation function based on the shape of the inverse covariance matrix.

        Returns
        -------
        exp_term_fn : Callable[[ndarray], ndarray]
            Function to compute the exponential term of the Gaussian likelihood.
        """
        if self.inv_covar.ndim == 2:
            return self._exponential_term_full
        elif self.inv_covar.ndim == 1 and self.inv_covar.size > 1:
            return self._exponential_term_diagonal
        elif self.inv_covar.ndim == 0 or self.inv_covar.size == 1:
            return self._exponential_term_scalar
        else:
            raise ValueError("Invalid shape for inverse covariance matrix.")

    def _exponential_term_full(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compute the exponential term of the Gaussian likelihood for a full covariance matrix.

        Parameters
        ----------
        residuals : ndarray, shape (batch, n)
            Residual vectors (observed_data - predicted_data).

        Returns
        -------
        exp_term : ndarray, shape (batch,)
            The exponential term values.
        """
        # Doing it this way avoids computing the full (batch x batch) matrix of residuals @ inv_covar @ residuals.T and then discarding all the off-diagonal terms
        return -0.5 * np.sum((residuals @ self.inv_covar) * residuals, axis=1)

    def _exponential_term_diagonal(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compute the exponential term of the Gaussian likelihood for a diagonal covariance matrix.

        Parameters
        ----------
        residuals : ndarray, shape (batch, n)
            Residual vectors (observed_data - predicted_data).

        Returns
        -------
        exp_term : ndarray, shape (batch,)
            The exponential term values.
        """
        return -0.5 * np.sum(residuals**2 * self.inv_covar[None, :], axis=1)

    def _exponential_term_scalar(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compute the exponential term of the Gaussian likelihood for a scalar covariance.

        Parameters
        ----------
        residuals : ndarray, shape (batch, n)
            Residual vectors (observed_data - predicted_data).

        Returns
        -------
        exp_term : ndarray, shape (batch,)
            The exponential term values.
        """
        return -0.5 * np.sum(residuals**2, axis=1) * self.inv_covar[0]


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
