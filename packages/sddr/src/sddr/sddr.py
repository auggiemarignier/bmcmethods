"""Functions for calculating the Savage-Dickey density ratio."""

from dataclasses import asdict, dataclass
from warnings import warn

import harmonic as hm
import numpy as np
from harmonic.model import RealNVPModel
from sampling.priors import CompoundPrior
from scipy.stats import gaussian_kde

from .marginalisation import marginalise_samples


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for training the flow model.

    Parameters to be passed to hm.model.FlowModel.fit().
    """

    batch_size: int = 64
    epochs: int = 10
    verbose: bool = True


@dataclass(frozen=True)
class RealNVPConfig:
    """Configuration of the RealNVP model.

    Just the list of parameters taken by hm.model.RealNVPModel.
    """

    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4
    learning_rate: float = 1e-3
    momentum: float = 0.9
    standardize: bool = False
    temperature: float = 0.8


class KDEModel:
    """Kernel Density Estimation model.

    Used as a fallback in the case where we marginalise down to 1D and RealNVP is not suitable.
    """

    def __init__(self, samples: np.ndarray) -> None:
        """Initialize the KDE model with given samples.

        Parameters
        ----------
        samples : ndarray, shape (num_samples, 1)
            Samples from the distribution to fit.
        """
        self.kde = gaussian_kde(samples.T)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict the log density at given points.

        Parameters
        ----------
        x : ndarray, shape (..., k)
            Points at which to evaluate the density.

        Returns
        -------
        log_density : ndarray, shape (...,)
            Log density values at the given points.
        """
        density = np.atleast_1d(self.kde(x.T))
        return np.log(density)


def fit_marginalised_posterior(
    samples: np.ndarray,
    marginal_indices: list[int],
    model_config: RealNVPConfig | None = None,
    train_config: TrainConfig | None = None,
) -> hm.model.FlowModel | KDEModel:
    """Fit a flow model to the marginalised posterior samples.

    Parameters
    ----------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters.
    marginal_indices : list of int
        Indices of the parameters to keep after marginalisation.
    model_config : RealNVPConfig, optional
        Configuration for the RealNVP model. If None, default configuration is used.
    train_config : TrainConfig, optional
        Configuration for training the flow model. If None, default configuration is used.

    Returns
    -------
    model : FlowModel | KDEModel
        Fitted flow/KDE model to the marginalised posterior.
    """
    marginalised_samples = marginalise_samples(samples, marginal_indices)

    if len(marginal_indices) == 1:
        warn(
            "Marginalising down to 1D; using KDEModel instead of RealNVPModel.",
            UserWarning,
            stacklevel=2,
        )
        return KDEModel(marginalised_samples)

    if model_config is None:
        model_config = RealNVPConfig()
    if train_config is None:
        train_config = TrainConfig()

    model = RealNVPModel(ndim_in=len(marginal_indices), **asdict(model_config))
    model.fit(X=marginalised_samples, **asdict(train_config))
    return model


def sddr(
    marginalised_posterior: hm.model.FlowModel | KDEModel,
    marginalised_prior: CompoundPrior,
    nu: np.ndarray,
) -> float:
    """Calculate the Savage-Dickey density ratio (SDDR) for given marginalised posterior and prior.

    Parameters
    ----------
    marginalised_posterior : FlowModel | KDEModel
        Fitted flow model or KDE model to the marginalised posterior.
    marginalised_prior : CompoundPrior
        Marginalised prior distribution.
    nu : ndarray, shape (k,)
        Point at which to evaluate the SDDR, where k is the number of marginalised parameters.

    Returns
    -------
    sddr : float
        Log SDDR value at the given point.
    """
    prior_log_prob = marginalised_prior(nu)
    posterior_log_prob = marginalised_posterior.predict(nu)
    return float(posterior_log_prob - prior_log_prob)
