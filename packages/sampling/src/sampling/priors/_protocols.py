"""Common Prior Protocols."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from .component import PriorComponent


class PriorType(StrEnum):
    """Enumeration of supported prior types."""

    GAUSSIAN = auto()
    UNIFORM = auto()


class PriorFunction(Protocol):
    """Protocol for prior functions.

    Stores configuration parameters for the prior.
    Helpful for marginalisation routines that need to access these parameters.
    """

    config_params: list[np.ndarray]
    n: int

    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """Calculate the log-prior for given model parameters.

        Parameters
        ----------
        model_params : ndarray
            Model parameters. Can be 1D for a single model or 2D for a batch.

        Returns
        -------
        log_prior : float | ndarray
            Log-prior value(s). Returns scalar for 1D input, array for 2D input.
        """

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (num_samples, n)
            Samples drawn from the prior.
        """


class PriorComponentConfig(Protocol):
    """Protocol for prior configuration objects."""

    type: PriorType
    indices: list[int]
    vectorised: bool

    def to_prior_component(self) -> PriorComponent:
        """Convert the configuration to a PriorComponent instance."""
