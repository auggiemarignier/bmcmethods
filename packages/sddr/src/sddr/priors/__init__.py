"""Prior definitions."""

from ._protocols import PriorFunction
from .component import PriorComponent
from .compound import CompoundPrior
from .gaussian import GaussianPrior
from .marginalisation import marginalise_prior
from .uniform import UniformPrior

__all__ = [
    "CompoundPrior",
    "GaussianPrior",
    "PriorComponent",
    "PriorFunction",
    "UniformPrior",
    "marginalise_prior",
]
