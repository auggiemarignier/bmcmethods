"""Triangular prior: sum of two independent Uniform(a, b) random variables.

This implements an independent triangular prior for each parameter, where the
triangular distribution for a single parameter is the convolution of two
identical uniform distributions on ``[a, b]``. For a single parameter the PDF
is

        pdf(x) =
                (x - 2a) / (b - a)**2   for 2a <= x <= a + b
                (2b - x) / (b - a)**2   for a + b <= x <= 2b

and zero outside ``[2a, 2b]``. For multivariate parameters the components are
assumed independent so the joint log-pdf is the sum of component log-pdfs.
"""

import numpy as np

from ._protocols import PriorType
from .component import PriorComponent


class TriangularPrior:
    """Triangular prior built as the sum of two independent Uniform(a, b).

    Parameters
    ----------
    lower_bounds : ndarray, shape (n,)
            Lower bounds ``a`` for each parameter's underlying uniform.
    upper_bounds : ndarray, shape (n,)
            Upper bounds ``b`` for each parameter's underlying uniform.
    """

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        lower_bounds = np.asarray(lower_bounds)
        upper_bounds = np.asarray(upper_bounds)
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError("lower_bounds and upper_bounds must have same shape")
        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Each lower bound must be less than the corresponding upper bound."
            )

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self._n = lower_bounds.size

        self._widths = self.upper_bounds - self.lower_bounds
        # Normalisation factor for a single component: (b-a)^2
        self._denom = self._widths**2

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        """Compute log-prior for given model parameters.

        Accepts 1D (shape (n,)) or 2D (shape (batch, n)) inputs. Returns a scalar
        (0D array) for 1D input, or 1D array of log-priors for batch input.
        """
        model_params = np.atleast_2d(model_params)
        # Broadcast bounds to batch shape
        a = self.lower_bounds
        b = self.upper_bounds
        denom = self._denom

        x = model_params
        left = 2 * a
        mid = a + b
        right = 2 * b

        log_pdf = np.full(x.shape, -np.inf)

        # Left segment: 2a <= x <= a+b
        mask_left = (x >= left) & (x <= mid)
        # pdf = (x - 2a) / (b - a)**2
        with np.errstate(divide="ignore"):
            pdf_left = (x - left) / denom
        log_pdf[mask_left] = np.log(pdf_left[mask_left])

        # Right segment: a+b <= x <= 2b
        mask_right = (x >= mid) & (x <= right)
        with np.errstate(divide="ignore"):
            pdf_right = (right - x) / denom
        log_pdf[mask_right] = np.log(pdf_right[mask_right])

        # Sum log-pdfs across components to get joint log-pdf
        log_priors = np.sum(log_pdf, axis=1)
        return log_priors.squeeze()

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        """Gradient of the log-prior with respect to model parameters.

        Gradient is computed component-wise. Where the log-prior is -inf (outside
        support) the gradient is set to 0.
        """
        was_1d = np.asarray(model_params).ndim == 1
        model_params = np.atleast_2d(model_params)
        a = self.lower_bounds
        b = self.upper_bounds

        x = model_params
        left = 2 * a
        mid = a + b
        right = 2 * b

        grads = np.zeros_like(x)

        # For left segment, d/dx log((x-2a)/w^2) = 1/(x - 2a)
        # Exclude x == 2a where the log-pdf is -inf and the expression would divide
        # by zero. Keep x == a+b included so the existing midpoint overwrite behavior
        # is preserved.
        mask_left = (x > left) & (x <= mid)
        # Use broadcasting to compute differences then index with the mask
        diff_left = x - left
        grads[mask_left] = 1.0 / diff_left[mask_left]

        # For right segment, d/dx log((2b-x)/w^2) = -1/(2b - x)
        # Exclude x == 2b where the log-pdf is -inf and the expression would divide
        # by zero. Keep x == a+b included so the existing midpoint overwrite behavior
        # is preserved.
        mask_right = (x >= mid) & (x < right)
        diff_right = right - x
        grads[mask_right] = -1.0 / diff_right[mask_right]

        # Keep zeros outside support (where log-prior is -inf)
        return grads.squeeze() if was_1d else grads

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the triangular prior by summing two independent uniforms."""
        u1 = rng.uniform(
            low=self.lower_bounds, high=self.upper_bounds, size=(num_samples, self._n)
        )
        u2 = rng.uniform(
            low=self.lower_bounds, high=self.upper_bounds, size=(num_samples, self._n)
        )
        return u1 + u2

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.lower_bounds, self.upper_bounds]

    @property
    def n(self) -> int:
        """Number of parameters in the Triangular prior."""
        return self._n


class TriangularPriorComponentConfig:
    """Configuration for a Triangular prior component."""

    type = PriorType.UNIFORM  # triangular arises from uniforms; choose UNIFORM type

    def __init__(
        self,
        lower_bounds: list[float] | np.ndarray,
        upper_bounds: list[float] | np.ndarray,
        indices: list[int],
    ) -> None:
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.indices = indices

    def to_prior_component(self) -> PriorComponent:
        """Build a PriorComponent from this config."""
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        prior_fn = TriangularPrior(lower_bounds=lower, upper_bounds=upper)
        return PriorComponent(type=self.type, prior_fn=prior_fn, indices=self.indices)
