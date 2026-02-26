"""Functions for calculating the Savage-Dickey density ratio."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, Self
from warnings import warn

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from sampling.priors import CompoundPrior

from .marginalisation import marginalise_samples

if TYPE_CHECKING:
    from harmonic.model import FlowModel


class TrainConfig(BaseModel):
    """Configuration for training the flow model.

    Parameters to be passed to hm.model.FlowModel.fit().
    """

    batch_size: int = 64
    epochs: int = 10
    verbose: bool = True

    model_config = ConfigDict(frozen=True)


class FlowModelConfig(ABC, BaseModel):
    """Base for model-specific configs exposing the model class.

    The heavy `harmonic` model classes are imported lazily inside `model_cls()`
    to avoid pulling JAX and friends at module import time.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def model_cls(self) -> type[FlowModel]:
        """Return the corresponding flow model class for this config.

        The return type is ``type[FlowModel]``; the concrete ``harmonic`` model
        classes are imported lazily to avoid requiring heavy ``harmonic`` (and
        its dependencies) at module import time.
        """
        ...


class RealNVPConfig(FlowModelConfig):
    """Configuration of the RealNVP model.

    Just the list of parameters taken by hm.model.RealNVPModel.
    """

    n_scaled_layers: int = 2
    n_unscaled_layers: int = 4

    def model_cls(self) -> type[FlowModel]:
        """Return the RealNVPModel class.

        Import the class lazily to avoid heavy imports during module import.
        """
        from harmonic.model import RealNVPModel

        return RealNVPModel


class RQSplineConfig(FlowModelConfig):
    """Configuration of the RQSpline model.

    Just the list of parameters taken by hm.model.RQSplineModel.
    """

    n_layers: int = 8
    n_bins: int = 8
    hidden_size: Sequence[int] = Field(default=(64, 64))
    spline_range: Sequence[float] = Field(default=(-10.0, 10.0))

    def model_cls(self) -> type[FlowModel]:
        """Return the RQSplineModel class.

        Import the class lazily to avoid heavy imports during module import.
        """
        from harmonic.model import RQSplineModel

        return RQSplineModel


default_model_configs = {
    "RealNVP": RealNVPConfig(),
    "RQSpline": RQSplineConfig(),
}


class FlowConfig(BaseModel):
    """Configuration of the flow model.

    Just the list of parameters taken by hm.model.FlowModel, plus a choice of flow type (RealNVP or RQSpline).
    """

    flow_type: Literal["RealNVP", "RQSpline"] = "RQSpline"
    flow_model_config: RQSplineConfig | RealNVPConfig = Field(
        default_factory=lambda: default_model_configs["RQSpline"]
    )
    standardize: bool = False
    learning_rate: float = 1e-3
    momentum: float = 0.9

    @computed_field
    def temperature(self) -> float:
        """Read-only temperature included in model dumps (not settable by user).

        `temperature` is different to the default in `harmonic` because for SDDR we don't want tempering.
        """
        return 1.0

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="after")
    def consistent_flow_type_and_model_config(self) -> Self:
        """Validate that flow_type and flow_model_config are consistent."""

        expected_config_type = type(default_model_configs[self.flow_type])
        actual_config_type = type(self.flow_model_config)
        if actual_config_type != expected_config_type:
            msg = (
                f"flow_type '{self.flow_type}' is inconsistent with flow_model_config type "
                f"{actual_config_type.__name__}. Expected {expected_config_type.__name__}."
            )
            raise ValueError(msg)
        return self


def fit_marginalised_posterior(
    samples: np.ndarray,
    marginal_indices: list[int],
    flow_config: FlowConfig | None = None,
    train_config: TrainConfig | None = None,
) -> FlowModel:
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

    if len(marginal_indices) == 1 and flow_config.flow_type == "RealNVP":
        warn(
            "Using RealNVP with a 1D marginal is not supported. Falling back to a default RQSpline model.",
            stacklevel=2,
        )
        flow_config = flow_config.model_copy(
            update={
                "flow_type": "RQSpline",
                "flow_model_config": default_model_configs["RQSpline"],
            }
        )

    marginalised_samples = marginalise_samples(samples, marginal_indices)

    if train_config is None:
        train_config = TrainConfig()

    model_cls = flow_config.flow_model_config.model_cls()
    flow_kwargs = flow_config.model_dump(exclude={"flow_model_config", "flow_type"})
    model_kwargs = flow_config.flow_model_config.model_dump()
    model = model_cls(ndim_in=len(marginal_indices), **model_kwargs, **flow_kwargs)
    model.fit(X=marginalised_samples, **train_config.model_dump())
    return model


def sddr(
    marginalised_posterior: FlowModel,
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
