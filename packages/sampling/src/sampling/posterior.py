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
        """
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn

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
