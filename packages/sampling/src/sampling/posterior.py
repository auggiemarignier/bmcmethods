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
        likelihood_fn: Callable[[np.ndarray], float],
        prior_fn: Callable[[np.ndarray], float],
    ) -> None:
        """
        Initialize the Posterior.

        Parameters
        ----------
        likelihood_fn : Callable[[np.ndarray], float]
            Likelihood function that takes model parameters and returns the log-likelihood.
        prior_fn : Callable[[np.ndarray], float]
            Prior function that takes model parameters and returns the log-prior.
        """
        self.likelihood_fn = likelihood_fn
        self.prior_fn = prior_fn

    def __call__(self, model_params: np.ndarray) -> float:
        """
        Evaluate the log-posterior for given model parameters.

        Parameters
        ----------
        model_params : ndarray
            Model parameters at which to evaluate the posterior.

        Returns
        -------
        log_posterior : float
            The log-posterior value.
        """
        log_likelihood = self.likelihood_fn(model_params)
        log_prior = self.prior_fn(model_params)
        return log_likelihood + log_prior
