"""Prior definitions."""

from ._protocols import PriorFunction, PriorType
from .component import PriorComponent
from .compound import CompoundPrior
from .gaussian import GaussianPrior
from .triangular import TriangularPrior
from .uniform import UniformPrior
from .wrapped import WrappedUniformPrior

__all__ = [
    "CompoundPrior",
    "GaussianPrior",
    "PriorComponent",
    "PriorFunction",
    "PriorType",
    "TriangularPrior",
    "UniformPrior",
    "WrappedUniformPrior",
]
