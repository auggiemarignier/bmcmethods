"""Uniform Prior."""

import numpy as np

from ._protocols import PriorType
from .component import PriorComponent


class UniformPrior:
    """Class representing a Uniform prior.

    Parameters
    ----------
    lower_bounds : ndarray, shape (n,)
        Lower bounds of the uniform prior.
    upper_bounds : ndarray, shape (n,)
        Upper bounds of the uniform prior.
    vectorised : bool, optional
        If True, the prior can evaluate batches of models (shape (batch_size, n)).
        If False, evaluates single models (shape (n,)). Default is True.

    Raises
    ------
    ValueError
        If `lower_bounds` and `upper_bounds` have mismatched shapes,
        or if any lower bound is not less than the corresponding upper bound.
    """

    def __init__(
        self,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        vectorised: bool = True,
    ) -> None:
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                f"Shape mismatch: lower_bounds has shape {lower_bounds.shape}, upper_bounds has shape {upper_bounds.shape}."
            )
        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Each lower bound must be less than the corresponding upper bound."
            )
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self._n = lower_bounds.size
        self._volume = np.prod(upper_bounds - lower_bounds)
        self._normalisation = -np.log(self._volume)
        self._call_fn = self._call_vectorised if vectorised else self._call_single

    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """Uniform log-prior.

        Parameters
        ----------
        model_params : ndarray
            If vectorised=False: shape (n,) for single model evaluation.
            If vectorised=True: shape (batch_size, n) for batch evaluation.

        Returns
        -------
        float or ndarray
            If vectorised=False: scalar log-prior value.
            If vectorised=True: array of shape (batch_size,) with log-prior values.
        """
        return self._call_fn(model_params)

    def _call_single(self, model_params: np.ndarray) -> float:
        """Uniform log-prior for a single model."""
        out_of_bounds = np.any(
            (model_params < self.lower_bounds) | (model_params > self.upper_bounds)
        )
        return -np.inf if out_of_bounds else self._normalisation

    def _call_vectorised(self, model_params: np.ndarray) -> np.ndarray:
        """Uniform log-prior for a batch of models.

        Parameters
        ----------
        model_params : ndarray, shape (batch_size, n)
            Batch of model parameters.

        Returns
        -------
        log_priors : ndarray, shape (batch_size,)
            Log-prior values for each model.
        """
        out_of_bounds = np.any(
            (model_params < self.lower_bounds) | (model_params > self.upper_bounds),
            axis=1,
        )
        log_priors = np.full(model_params.shape[0], self._normalisation)
        log_priors[out_of_bounds] = -np.inf
        return log_priors

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the Uniform prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (num_samples, n)
            Samples drawn from the Uniform prior.
        """
        return rng.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(num_samples, self._n),
        )

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.lower_bounds, self.upper_bounds]

    @property
    def n(self) -> int:
        """Number of parameters in the Uniform prior."""
        return self._n

    @property
    def volume(self) -> float:
        """Volume of the Uniform prior."""
        return self._volume


class UniformPriorComponentConfig:
    """Configuration for a Uniform prior component."""

    type = PriorType.UNIFORM

    def __init__(
        self,
        lower_bounds: list[float] | np.ndarray,
        upper_bounds: list[float] | np.ndarray,
        indices: list[int],
        vectorised: bool = True,
    ) -> None:
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.indices = indices
        self.vectorised = vectorised

    def to_prior_component(self) -> PriorComponent:
        """Build a PriorComponent from this config."""
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        prior_fn = UniformPrior(
            lower_bounds=lower, upper_bounds=upper, vectorised=self.vectorised
        )

        return PriorComponent(
            prior_fn=prior_fn, indices=self.indices, vectorised=self.vectorised
        )
