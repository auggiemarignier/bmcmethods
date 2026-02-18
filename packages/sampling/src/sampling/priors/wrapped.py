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
        try:
            self.type = PriorType(type.lower())
        except ValueError:
            raise ValueError(f"Unknown prior type: {type}")

        prior_cls = _AVAILABLE_PRIORS[self.type]
        self._base = prior_cls(**kwargs)

    def _wrap(self, model_params: np.ndarray) -> np.ndarray:
        """Wrap model parameters around the specified bounds.

        Parameters
        ----------
        model_params : ndarray
            Model parameters to be wrapped. Can be 1D for a single model or 2D for a batch.

        Returns
        -------
        wrapped_params : ndarray
            Wrapped model parameters, with the same shape as the input.
        """
        wrapped_params = model_params.copy()
        for i, (lower, upper) in enumerate(self.wrap_bounds):
            range_width = upper - lower
            wrapped_params[..., i] = (
                wrapped_params[..., i] - lower
            ) % range_width + lower
        return wrapped_params

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
        wrapped_params = self._wrap(model_params)
        return self._base(wrapped_params)
