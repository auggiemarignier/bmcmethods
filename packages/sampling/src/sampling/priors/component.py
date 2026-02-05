"""Prior component combining a prior function with parameter indices."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ._protocols import PriorFunction
from ._utils import _normalise_indices


@dataclass
class PriorComponent:
    """Class representing a prior component.

    Multiple prior components can be combined to form a joint prior over
    different subsets of model parameters.

    Parameters
    ----------
    prior_fn : PriorFunction
        Prior function that takes model parameters and returns the log-prior.
    indices : Sequence[int] | slice | np.ndarray
        Indices of the model parameters that this prior component applies to.
        Will be converted to a sorted numpy array internally.
    """

    prior_fn: PriorFunction
    indices: Sequence[int] | slice | np.ndarray

    def __post_init__(self) -> None:
        """Convert indices to a normalized numpy array."""
        object.__setattr__(
            self, "indices", _normalise_indices(self.indices, self.prior_fn.n)
        )

    @property
    def n(self) -> int:
        """Number of parameters in this prior component."""
        return int(self.indices.size)
