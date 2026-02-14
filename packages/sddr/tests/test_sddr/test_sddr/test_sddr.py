"""Tests for the SDDR calculation functions."""

from dataclasses import FrozenInstanceError, fields
from unittest.mock import MagicMock

import numpy as np
import pytest
from harmonic.model import FlowModel, RealNVPModel, RQSplineModel
from sampling.priors import CompoundPrior, GaussianPrior, PriorComponent
from sddr.sddr import (
    FlowConfig,
    RealNVPConfig,
    RQSplineConfig,
    TrainConfig,
    fit_marginalised_posterior,
    sddr,
)


class TestFitMarginalisedPosterior:
    """Tests for fit_marginalised_posterior function."""

    @pytest.fixture
    def samples(self, rng: np.random.Generator) -> np.ndarray:
        """Sample MCMC data for testing."""
        return rng.standard_normal((100, 5))

    def test_default_configs(self, samples: np.ndarray) -> None:
        """Test that default configs are used when None is provided."""
        marginal_indices = [0, 1]

        model = fit_marginalised_posterior(samples, marginal_indices)

        assert model is not None
        assert isinstance(model, RQSplineModel)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_flow_choice(self, samples: np.ndarray) -> None:
        """Test that the right type of flow model is returned."""
        marginal_indices = [0, 1]
        flow_config = FlowConfig(flow_type="RealNVP")

        model = fit_marginalised_posterior(
            samples, marginal_indices, flow_config=flow_config
        )

        assert isinstance(model, RealNVPModel)

    def test_custom_train_config(self, samples: np.ndarray) -> None:
        """Test that custom train config is used."""
        marginal_indices = [0, 1]
        train_config = TrainConfig(epochs=5, batch_size=32, verbose=False)

        model = fit_marginalised_posterior(
            samples, marginal_indices, train_config=train_config
        )

        assert isinstance(model, FlowModel)

    def test_marginalisation(self, samples: np.ndarray) -> None:
        """Test that samples are correctly marginalised."""
        marginal_indices = [1, 3]

        model = fit_marginalised_posterior(samples, marginal_indices)

        # Model should be fitted to 2D marginalised samples
        assert isinstance(model, FlowModel)
        assert model.ndim == 2

    def test_single_parameter_marginalisation(self, samples: np.ndarray) -> None:
        """Test marginalisation to a single parameter returns a RQSplineModel even if RealNVP is requested."""

        marginal_indices = [2]

        flow_config = FlowConfig(flow_type="RealNVP")
        with pytest.warns(UserWarning, match="1D"):
            model = fit_marginalised_posterior(
                samples, marginal_indices, flow_config=flow_config
            )

        assert isinstance(model, RQSplineModel)

    def test_all_parameters(self, samples: np.ndarray) -> None:
        """Test keeping all parameters (no marginalisation)."""
        marginal_indices = [0, 1, 2, 3, 4]

        model = fit_marginalised_posterior(samples, marginal_indices)

        assert isinstance(model, FlowModel)
        assert model.ndim == 5


class TestSDDR:
    """Tests for the sddr function."""

    def test_sddr_calculation(self) -> None:
        """Test basic SDDR calculation."""
        # Create a simple prior
        prior = CompoundPrior(
            [PriorComponent(GaussianPrior(np.zeros(2), np.eye(2)), np.arange(2))]
        )

        # Mock the posterior model
        mock_posterior = MagicMock()
        mock_posterior.predict.return_value = -2.5

        # Evaluate SDDR at the prior mean
        nu = np.zeros(2)
        result = sddr(mock_posterior, prior, nu)

        # Check that the function called predict
        mock_posterior.predict.assert_called_once()
        assert isinstance(result, float)

    def test_sddr_at_prior_mean(self) -> None:
        """Test SDDR at the prior mean."""
        # Create a Gaussian prior
        mean = np.array([1.0, 2.0])
        covar = np.eye(2)
        prior = CompoundPrior(
            [PriorComponent(GaussianPrior(mean, covar), np.arange(2))]
        )

        # Mock posterior that returns the same as prior
        mock_posterior = MagicMock()
        prior_log_prob = prior(mean)
        mock_posterior.predict.return_value = prior_log_prob

        result = sddr(mock_posterior, prior, mean)

        # SDDR should be zero (log(1) = 0) when prior and posterior agree
        assert np.isclose(result, 0.0)

    def test_sddr_with_different_values(self) -> None:
        """Test SDDR with different prior and posterior probabilities."""
        prior = CompoundPrior(
            [PriorComponent(GaussianPrior(np.zeros(2), np.eye(2)), np.arange(2))]
        )

        mock_posterior = MagicMock()
        mock_posterior.predict.return_value = -3.0

        nu = np.array([0.5, 0.5])
        result = sddr(mock_posterior, prior, nu)

        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_sddr_returns_float(self) -> None:
        """Test that SDDR returns a Python float, not numpy scalar."""
        prior = CompoundPrior(
            [PriorComponent(GaussianPrior(np.zeros(1), np.eye(1)), np.array([0]))]
        )

        mock_posterior = MagicMock()
        mock_posterior.predict.return_value = -1.5

        nu = np.array([0.0])
        result = sddr(mock_posterior, prior, nu)

        assert isinstance(result, float)
        assert not isinstance(result, np.floating)


def test_RealNVPConfig_model_cls() -> None:
    """Test that RealNVPConfig returns the correct model class."""
    config = RealNVPConfig()
    assert config.model_cls() == RealNVPModel


def test_RQSplineConfig_model_cls() -> None:
    """Test that RQSplineConfig returns the correct model class."""
    config = RQSplineConfig()
    assert config.model_cls() == RQSplineModel


def test_FlowConfig_cannot_initialise_temperature() -> None:
    """Test that temperature cannot be set in FlowConfig."""
    with pytest.raises(TypeError):
        FlowConfig(temperature=0.5)


def test_FlowConfig_temperature_is_fixed() -> None:
    """Test that temperature is fixed to 1.0 in FlowConfig."""
    config = FlowConfig()
    assert config.temperature == 1.0


@pytest.mark.parametrize(
    "config_class", [RealNVPConfig, RQSplineConfig, FlowConfig, TrainConfig]
)
def test_config_classes_are_frozen(config_class) -> None:
    """Test that config dataclasses are frozen."""
    config = config_class()
    field_names = [field.name for field in fields(config_class)]
    for field_name in field_names:
        with pytest.raises(FrozenInstanceError):
            setattr(config, field_name, 123)
