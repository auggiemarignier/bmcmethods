"""Base class for Likelihood objects.

The focus is the separation of data from the callable so that things are nicely pickleable.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import numpy as np

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
    def from_state(cls, state: T) -> Self:
        """Rebuild the forward function from a serialisable state object."""

    @abstractmethod
    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """Forward modelling."""


class ForwardGradientBase[T](ABC):
    """Generic interface for forward gradient functions that may require internal data.

    The heavy data should be stored in a separate state.

    The user is expected to use this class in their applications to separate state and callables.
    """

    state: T

    @classmethod
    @abstractmethod
    def from_state(cls, state: T) -> Self:
        """Rebuild the forward gradient function from a serialisable state object."""

    @abstractmethod
    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """Forward modelling gradient."""


class NoForwardGradient(ForwardGradientBase[None]):
    """Internal no-op sentinel used when no forward gradient is available.

    This implements the same minimal interface as other `ForwardGradientBase`
    implementations but raises a runtime error when called. Placing this
    sentinel in the base module avoids import-time cycles between the
    worker initialisation and individual likelihood modules.
    """

    state = None

    @classmethod
    def from_state(cls, state: None) -> "NoForwardGradient":
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        raise RuntimeError("A gradient for the forward function is needed.")


@dataclass(frozen=True)
class IdentityState:
    """Empty serialisable state for the identity forward model."""
    pass


class IdentityForward(ForwardBase[IdentityState]):
    """Internal no-op forward placed in the base module for consistency.

    Returns the input parameters unchanged — used when no forward model is
    provided so that workers and likelihoods can uniformly expect a
    `ForwardBase` implementation.
    """

    state = IdentityState()

    @classmethod
    def from_state(cls, state: IdentityState) -> "IdentityForward":
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        return model_params


class LikelihoodBase[T](ABC):
    """
    Generic interface for a model likelihood.

    The important bit is: the sampler should call __call__(theta), while the heavy state lives separately in self.state.
    """

    state: T
    forward: ForwardBase | None = None
    forward_gradient: ForwardGradientBase | None = None

    @classmethod
    @abstractmethod
    def from_state(
        cls,
        state: T,
        *,
        forward: ForwardBase | None = None,
        forward_gradient: ForwardGradientBase | None = None,
    ) -> Self:
        """Rebuild the likelihood from a serialisable state object."""

    @abstractmethod
    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """Return the log likelihood."""
