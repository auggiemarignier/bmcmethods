"""Uniform Prior."""

import numpy as np

from ._protocols import PriorType
from .component import PriorComponent


class UniformPrior:
    """Class representing a Uniform prior.

    Parameters
    ----------
    lower_bounds : ndarray, shape (n,)
        Lower bounds of the uniform prior.
    upper_bounds : ndarray, shape (n,)
        Upper bounds of the uniform prior.

    Raises
    ------
    ValueError
        If `lower_bounds` and `upper_bounds` have mismatched shapes,
        or if any lower bound is not less than the corresponding upper bound.
    """

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                f"Shape mismatch: lower_bounds has shape {lower_bounds.shape}, upper_bounds has shape {upper_bounds.shape}."
            )
        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Each lower bound must be less than the corresponding upper bound."
            )
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self._n = lower_bounds.size

    def __call__(self, model_params: np.ndarray) -> float:
        """Uniform log-prior."""
        out_of_bounds = np.any(
            (model_params < self.lower_bounds) | (model_params > self.upper_bounds)
        )
        return float(np.where(out_of_bounds, -np.inf, 0.0))

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.lower_bounds, self.upper_bounds]

    @property
    def n(self) -> int:
        """Number of parameters in the Uniform prior."""
        return self._n


class UniformPriorComponentConfig:
    """Configuration for a Uniform prior component."""

    type = PriorType.UNIFORM

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
        prior_fn = UniformPrior(lower_bounds=lower, upper_bounds=upper)

        return PriorComponent(prior_fn=prior_fn, indices=self.indices)
