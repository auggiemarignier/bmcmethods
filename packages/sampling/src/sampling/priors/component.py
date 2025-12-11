"""Prior component combining a prior function with parameter indices."""

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field

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
    """

    prior_fn: PriorFunction
    indices: InitVar[Sequence[int] | slice | np.ndarray]

    _indices: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self, indices: Sequence[int] | slice | np.ndarray) -> None:
        """Convert indices to a numpy array if it's a slice."""
        self._indices = _normalise_indices(indices, self.prior_fn.n)

    @property
    def n(self) -> int:
        """Number of parameters in this prior component."""
        return int(np.asarray(self.indices).size)

    @property
    def indices(self) -> np.ndarray:
        """Get the indices as a numpy array."""
        return self._indices
