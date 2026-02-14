"""Functions for calculating the Savage-Dickey density ratio."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal
from warnings import warn

import harmonic as hm
import numpy as np
from harmonic.model import RealNVPModel, RQSplineModel
from sampling.priors import CompoundPrior

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
class ModelConfig(ABC):
    """Base for model-specific configs exposing the model class."""

    @abstractmethod
    def model_cls(self) -> Any:
        """Return the corresponding model class for this config."""
        ...


@dataclass(frozen=True)
class RealNVPConfig(ModelConfig):
    """Configuration of the RealNVP model.

    Just the list of parameters taken by hm.model.RealNVPModel.
    """

    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4

    def model_cls(self) -> Any:
        """Return the RealNVPModel class."""
        return RealNVPModel


@dataclass(frozen=True)
class RQSplineConfig(ModelConfig):
    """Configuration of the RQSpline model.

    Just the list of parameters taken by hm.model.RQSplineModel.
    """

    n_layers: int = 8
    n_bins: int = 8
    hidden_size: Sequence[int] = (64, 64)
    spline_range: Sequence[float] = (-10.0, 10.0)

    def model_cls(self) -> Any:
        """Return the RQSplineModel class."""
        return RQSplineModel


@dataclass(frozen=True)
class FlowConfig:
    """Configuration of the flow model.

    Just the list of parameters taken by hm.model.FlowModel, plus a choice of flow type (RealNVP or RQSpline).

    `temperature` is different to the default in `harmonic` because for SDDR we don't want tempering.
    """

    flow_type: Literal["RealNVP", "RQSpline"] = "RQSpline"
    model_config: ModelConfig | None = None
    standardize: bool = False
    learning_rate: float = 1e-3
    momentum: float = 0.9
    temperature: float = field(default=1.0, init=False)  # No tempering for SDDR


default_model_configs = {
    "RealNVP": RealNVPConfig(),
    "RQSpline": RQSplineConfig(),
}


def fit_marginalised_posterior(
    samples: np.ndarray,
    marginal_indices: list[int],
    flow_config: FlowConfig | None = None,
    train_config: TrainConfig | None = None,
) -> hm.model.FlowModel:
    """Fit a flow model to the marginalised posterior samples.

    Parameters
    ----------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters.
    marginal_indices : list of int
        Indices of the parameters to keep after marginalisation.
    flow_config : FlowConfig, optional
        Configuration for the flow model. If None, default configuration is used.
    train_config : TrainConfig, optional
        Configuration for training the flow model. If None, default configuration is used.

    Returns
    -------
    model : FlowModel
        Fitted flow model to the marginalised posterior.
    """
    if flow_config is None:
        flow_config = FlowConfig()

    # default model_config from flow_type if not provided
    if flow_config.model_config is None:
        flow_config = replace(
            flow_config, model_config=default_model_configs[flow_config.flow_type]
        )

    if len(marginal_indices) == 1 and flow_config.flow_type == "RealNVP":
        warn(
            "Using RealNVP with a 1D marginal is not supported. Falling back to a default RQSpline model.",
            stacklevel=2,
        )
        flow_config = replace(
            flow_config,
            flow_type="RQSpline",
            model_config=default_model_configs["RQSpline"],
        )

    marginalised_samples = marginalise_samples(samples, marginal_indices)

    if train_config is None:
        train_config = TrainConfig()

    model_cls = flow_config.model_config.model_cls()
    flow_cfg = asdict(flow_config)
    model_cfg = flow_cfg.pop("model_config")
    _ = flow_cfg.pop("flow_type")
    model = model_cls(ndim_in=len(marginal_indices), **model_cfg, **flow_cfg)
    model.fit(X=marginalised_samples, **asdict(train_config))
    return model


def sddr(
    marginalised_posterior: hm.model.FlowModel,
    marginalised_prior: CompoundPrior,
    nu: np.ndarray,
) -> float:
    """Calculate the Savage-Dickey density ratio (SDDR) for given marginalised posterior and prior.

    Parameters
    ----------
    marginalised_posterior : FlowModel
        Fitted flow model to the marginalised posterior.
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
