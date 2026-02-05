"""Gaussian Prior."""

import numpy as np

from ._protocols import PriorType
from .component import PriorComponent


class GaussianPrior:
    """Class representing a Gaussian prior.

    Parameters
    ----------
    mean : ndarray, shape (n,)
        Mean of the Gaussian prior e.g. a reference model.
    inv_covar : ndarray, shape (n, n)
        Inverse Covariance matrix of the Gaussian prior.
    vectorised : bool, optional
        If True, the prior can evaluate batches of models (shape (batch_size, n)).
        If False, evaluates single models (shape (n,)). Default is True.

    Raises
    ------
    ValueError
        If the inverse covariance matrix is not symmetric, not positive semidefinite, or has a shape mismatch with the mean.
    """

    def __init__(
        self, mean: np.ndarray, inv_covar: np.ndarray, vectorised: bool = False
    ) -> None:
        _validate_covariance_matrix(inv_covar, mean.size)
        self.mean = mean
        self.inv_covar = inv_covar
        self._n = mean.size

        self._normalisation = (
            -0.5 * self._n * np.log(2 * np.pi)
            + 0.5 * np.linalg.slogdet(self.inv_covar)[1]
        )
        self._call_fn = self._call_vectorised if vectorised else self._call_single

    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """Gaussian log-prior.

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
        """Gaussian log-prior for a single model."""
        diff = model_params - self.mean
        return float(-0.5 * diff.T @ self.inv_covar @ diff) + self._normalisation

    def _call_vectorised(self, model_params: np.ndarray) -> np.ndarray:
        """Gaussian log-prior for a batch of models.

        Parameters
        ----------
        model_params : ndarray, shape (batch_size, n)
            Batch of model parameters.

        Returns
        -------
        log_priors : ndarray, shape (batch_size,)
            Log-prior values for each model.
        """
        diff = model_params - self.mean  # shape (batch_size, n)
        # Compute quadratic form for each model: diff @ inv_covar @ diff.T
        # More efficient: (diff @ inv_covar * diff).sum(axis=1)
        log_priors = (
            -0.5 * np.sum(diff @ self.inv_covar * diff, axis=1) + self._normalisation
        )
        return log_priors

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the Gaussian prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (num_samples, n)
            Samples drawn from the Gaussian prior.
        """
        covar = np.linalg.inv(self.inv_covar)
        return rng.multivariate_normal(self.mean, covar, size=num_samples)

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.mean, self.inv_covar]

    @property
    def n(self) -> int:
        """Number of parameters in the Gaussian prior."""
        return self._n


def _validate_covariance_matrix(covar: np.ndarray, N: int) -> None:
    """
    Validate that the inverse covariance matrix is symmetric and positive semidefinite.

    Parameters
    ----------
    covar : ndarray, shape (n, n)
        Inverse covariance matrix to validate.
    N : int
        Expected size of the inverse covariance matrix.

    Raises
    ------
    ValueError
        If the inverse covariance matrix
            - has incorrect shape;
            - is not symmetric; or
            - is not positive semidefinite.
    """
    if covar.shape != (N, N):
        raise ValueError(f"Covariance matrix must be of shape ({N}, {N}).")

    if not np.allclose(covar, covar.T):
        raise ValueError("Covariance matrix must be symmetric.")

    try:
        np.linalg.cholesky(covar)
        # If Cholesky decomposition succeeds, the matrix is positive definite
        # It is very unlikely for a realistic inverse covariance matrix to have zero eigenvalues (positive semidefinite) so this check is sufficient
    except np.linalg.LinAlgError as e:
        raise ValueError(
            "Inverse covariance matrix must be positive semidefinite."
        ) from e


class GaussianPriorComponentConfig:
    """Configuration for a Gaussian prior component."""

    type = PriorType.GAUSSIAN

    def __init__(
        self,
        mean: list[float] | np.ndarray,
        inv_covar: list[list[float]] | np.ndarray,
        indices: list[int],
        vectorised: bool = True,
    ) -> None:
        self.mean = mean
        self.inv_covar = inv_covar
        self.indices = indices
        self.vectorised = vectorised

    def to_prior_component(self) -> PriorComponent:
        """Build a PriorComponent from this config."""
        mean = np.asarray(self.mean)
        inv_covar = np.asarray(self.inv_covar)
        prior_fn = GaussianPrior(
            mean=mean, inv_covar=inv_covar, vectorised=self.vectorised
        )

        return PriorComponent(
            prior_fn=prior_fn, indices=self.indices, vectorised=self.vectorised
        )
