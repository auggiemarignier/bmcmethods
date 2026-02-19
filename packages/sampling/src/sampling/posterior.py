"""Constructing and fitting the normalised posterior distribution."""

from collections.abc import Callable

import numpy as np


class Posterior:
    """
    Represents the posterior distribution for MCMC sampling.

    The posterior distribution combines a likelihood function and a prior function
    to evaluate the log-posterior for given model parameters.
    """

    def __init__(
        self,
        likelihood_fn: Callable[[np.ndarray], float | np.ndarray],
        prior_fn: Callable[[np.ndarray], float | np.ndarray],
        likelihood_gradient_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        prior_gradient_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """
        Initialise the Posterior.

        Parameters
        ----------
        likelihood_fn : Callable[[np.ndarray], float | np.ndarray]
            Likelihood function that takes model parameters and returns the log-likelihood.
            Can accept single models (1D array) or batched models (2D array).
        prior_fn : Callable[[np.ndarray], float | np.ndarray]
            Prior function that takes model parameters and returns the log-prior.
            Can accept single models (1D array) or batched models (2D array).
        likelihood_gradient_fn : Callable[[np.ndarray], np.ndarray] | None, optional
            Gradient of the likelihood function with respect to model parameters.
            Should return an array of the same shape as model parameters.
            If None, the gradient method will raise an error if called.
        prior_gradient_fn : Callable[[np.ndarray], np.ndarray] | None, optional
            Gradient of the prior function with respect to model parameters.
            Should return an array of the same shape as model parameters.
            If None, the gradient method will raise an error if called.
        """
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn
        self.likelihood_gradient_fn = likelihood_gradient_fn
        self.prior_gradient_fn = prior_gradient_fn

        # Setting the gradient function based on the availability of likelihood and prior gradients
        # Doing it here avoids checking for gradients every time the gradient method is called, improving efficiency
        if likelihood_gradient_fn is None or prior_gradient_fn is None:
            self.gradient_fn = self._gradient_not_available
        else:
            self.gradient_fn = self._gradient

    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """
        Evaluate the log-posterior for given model parameters.

        Parameters
        ----------
        model_params : ndarray
            Model parameters at which to evaluate the posterior.
            Shape (n_params,) for a single model or (batch_size, n_params) for batched models.

        Returns
        -------
        log_posterior : float | ndarray
            The log-posterior value. Returns a scalar for single models or
            an array of shape (batch_size,) for batched models.
        """
        log_likelihood = self.likelihood_fn(model_params)
        log_prior = self.prior_fn(model_params)
        return log_likelihood + log_prior

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        """
        Evaluate the gradient of the log-posterior for given model parameters.

        Parameters
        ----------
        model_params : ndarray
            Model parameters at which to evaluate the gradient.
            Shape (n_params,) for a single model or (batch_size, n_params) for batched models.

        Returns
        -------
        gradient : ndarray
            The gradient of the log-posterior with respect to the model parameters.
            Shape (n_params,) for a single model or (batch_size, n_params) for batched models.

        Raises
        ------
        ValueError
            If either the likelihood or prior gradient function is not provided.
        """
        return self.gradient_fn(model_params)

    def _gradient(self, model_params: np.ndarray) -> np.ndarray:
        """Compute the gradient of the log-posterior."""

        # Letting type checkers know these attributes will be set if this method is called
        self.likelihood_gradient_fn: Callable[[np.ndarray], np.ndarray]
        self.prior_gradient_fn: Callable[[np.ndarray], np.ndarray]

        likelihood_grad = self.likelihood_gradient_fn(model_params)
        prior_grad = self.prior_gradient_fn(model_params)
        return likelihood_grad + prior_grad

    def _gradient_not_available(self, model_params: np.ndarray) -> np.ndarray:
        """Raise an error if gradient functions are not provided."""
        raise ValueError(
            "Gradient functions for likelihood and prior must be provided to compute the posterior gradient."
        )
