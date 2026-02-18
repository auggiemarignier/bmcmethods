"""Wrapping uniform priors around its bounds."""

import numpy as np

from sampling.priors import PriorComponent
from sampling.priors._protocols import PriorType
from sampling.priors.uniform import UniformPrior


class WrappedUniformPrior(UniformPrior):
    """A uniform prior that wraps parameters around its bounds."""

    def __init__(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> None:
        """
        Initialize the WrappedUniformPrior.

        Parameters
        ----------
        lower_bounds : numpy array
            Lower bounds for each parameter to be wrapped.
        upper_bounds : numpy array
            Upper bounds for each parameter to be wrapped.
        """
        self._upper_bounds = upper_bounds
        self._lower_bounds = lower_bounds
        self._bounds_widths = self._upper_bounds - self._lower_bounds

        super().__init__(
            lower_bounds=self._lower_bounds, upper_bounds=self._upper_bounds
        )

    def _wrap(self, model_params: np.ndarray) -> np.ndarray:
        """Wrap model parameters around the specified bounds.

        w = (x - lower) % width + lower

        Parameters
        ----------
        model_params : ndarray
            Model parameters to be wrapped. Can be 1D for a single model or 2D for a batch.

        Returns
        -------
        wrapped_params : ndarray
            Wrapped model parameters, with the same shape as the input.
        """
        return (
            model_params - self._lower_bounds
        ) % self._bounds_widths + self._lower_bounds

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """Calculate the log-prior for given model parameters, after wrapping.

        Parameters
        ----------
        model_params : ndarray
            Model parameters. Can be 1D for a single model or 2D for a batch.

        Returns
        -------
        log_prior : ndarray
            Log-prior value(s) from the base prior, evaluated at the wrapped parameters.
            Returns scalar (0D array) for 1D input, array for 2D input.
        """
        return super().__call__(self._wrap(model_params))

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the wrapped prior.

        In practice this just samples from the base prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (num_samples, n)
            Samples drawn from the wrapped prior.
        """
        return super().sample(num_samples, rng)


class WrappedUniformPriorComponentConfig:
    """Configuration for a Wrapped Uniform prior component."""

    type = PriorType.WRAPPED_UNIFORM

    def __init__(
        self,
        lower_bounds: list[float] | np.ndarray,
        upper_bounds: list[float] | np.ndarray,
        indices: list[int],
    ) -> None:
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.indices = indices

    def to_prior_component(self) -> PriorComponent:
        """Build a PriorComponent from this config."""
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        prior_fn = WrappedUniformPrior(lower_bounds=lower, upper_bounds=upper)

        return PriorComponent(type=self.type, prior_fn=prior_fn, indices=self.indices)
