"""Wrapping priors around specified bounds."""

from typing import Any

import numpy as np

from sampling.priors import PriorFunction
from sampling.priors._protocols import PriorType
from sampling.priors.gaussian import GaussianPrior
from sampling.priors.uniform import UniformPrior

_AVAILABLE_PRIORS: dict[PriorType, type[PriorFunction]] = {
    PriorType.GAUSSIAN: GaussianPrior,
    PriorType.UNIFORM: UniformPrior,
}


class WrappedPrior:
    """A prior that wraps parameters around specified bounds."""

    def __init__(
        self,
        wrap_bounds: list[tuple[float, float]],
        type: str | PriorType,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the WrappedPrior.

        Parameters
        ----------
        wrap_bounds : list of tuples
            List of (lower_bound, upper_bound) for each parameter to be wrapped.
        type : str or PriorType
            The type of prior to be wrapped.
        """
        self.wrap_bounds = wrap_bounds
        self._upper_bounds = np.array([b[1] for b in wrap_bounds])
        self._lower_bounds = np.array([b[0] for b in wrap_bounds])
        self._bounds_widths = self._upper_bounds - self._lower_bounds
        try:
            self.type = PriorType(type.lower())
        except ValueError:
            raise ValueError(f"Unknown prior type: {type}")

        prior_cls = _AVAILABLE_PRIORS[self.type]
        self._base = prior_cls(**kwargs)

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
        return self._base(self._wrap(model_params))

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
        return self._base.sample(num_samples, rng)
