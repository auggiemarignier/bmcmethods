"""Core component types for linear-Gaussian inference."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

type NDArrayFloat = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianComponent:
    """A single Gaussian component in the linear-Gaussian model.

    This is a frozen dataclass and the internal arrays are read-only for caching safety.

    Parameters
    ----------
    A : ndarray, shape (M, N_j)
        Linear mapping from the N_j-dimensional latent variable to the
        M-dimensional observation space.
    mu : ndarray, shape (N_j,)
        Prior mean of the latent variable.
    C : ndarray, shape (N_j, N_j)
        Prior covariance of the latent variable.
    """

    A: NDArrayFloat
    mu: NDArrayFloat
    C: NDArrayFloat

    def __post_init__(self) -> None:
        """Validate Gaussian component array dimensions and compatibility."""

        A = np.array(self.A, copy=True)
        mu = np.array(self.mu, copy=True)
        C = np.array(self.C, copy=True)

        if A.ndim != 2:
            raise ValueError("A must be a 2D array")
        if mu.ndim != 1:
            raise ValueError("mu must be a 1D array")
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C must be a 2D square matrix")
        if mu.shape[0] != C.shape[0]:
            raise ValueError("mu and C dimensions are incompatible")
        if A.shape[1] != mu.shape[0]:
            raise ValueError("A columns and mu are incompatible")

        A.setflags(write=False)
        mu.setflags(write=False)
        C.setflags(write=False)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "C", C)
