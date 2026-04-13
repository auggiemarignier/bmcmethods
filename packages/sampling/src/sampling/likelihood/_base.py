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


class ForwardBase[T](ABC):
    """Generic interface for forward functions that may require internal data.

    The heavy data should be stored in a separate state.

    The user is expected to use this class in their applications to separate state and callables.
    """

    state: T

    @classmethod
    @abstractmethod
    def from_state(cls, state: T) -> "ForwardBase[T]":
        """Rebuild the forward function from a seraialisable state object."""

    @abstractmethod
    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """Forward modelling."""
