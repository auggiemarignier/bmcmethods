"""Base class for Likelihood objects.

The focus is the separation of data from the callable so that things are nicely pickleable.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np


class LikelihoodBase[T](ABC):
    """
    Generic interface for a model likelihood.

    The important bit is: the sampler should call __call__(theta), while the heavy state lives separately in self.state.
    """

    state: T

    @classmethod
    @abstractmethod
    def from_state(cls, state: T) -> "LikelihoodBase[T]":
        """Rebuild the likelihood from a serialisable state object."""

    @abstractmethod
    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """Return the log likelihood."""


type ForwardFunction = Callable[[np.ndarray], np.ndarray]
type ForwardGradientFunction = Callable[[np.ndarray], np.ndarray]
