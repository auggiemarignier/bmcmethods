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

    def __call__(self, model_params: np.ndarray) -> float:
        """Calculate the log-prior for given model parameters."""


class PriorComponentConfig(Protocol):
    """Protocol for prior configuration objects."""

    type: PriorType
    indices: list[int]

    def to_prior_component(self) -> PriorComponent:
        """Convert the configuration to a PriorComponent instance."""
