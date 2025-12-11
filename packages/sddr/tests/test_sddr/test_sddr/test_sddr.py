"""Tests for the SDDR calculation functions."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from sampling.priors import CompoundPrior, GaussianPrior, PriorComponent
from sddr.sddr import (
    KDEModel,
    RealNVPConfig,
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
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_custom_model_config(self, samples: np.ndarray) -> None:
        """Test that custom model config is used."""
        marginal_indices = [0, 1]
        model_config = RealNVPConfig(n_scaled_layers=3, learning_rate=1e-4)

        model = fit_marginalised_posterior(
            samples, marginal_indices, model_config=model_config
        )

        assert model is not None

    def test_custom_train_config(self, samples: np.ndarray) -> None:
        """Test that custom train config is used."""
        marginal_indices = [0, 1]
        train_config = TrainConfig(epochs=5, batch_size=32, verbose=False)

        model = fit_marginalised_posterior(
            samples, marginal_indices, train_config=train_config
        )

        assert model is not None

    def test_marginalisation(self, samples: np.ndarray) -> None:
        """Test that samples are correctly marginalised."""
        from harmonic.model import FlowModel

        marginal_indices = [1, 3]

        model = fit_marginalised_posterior(samples, marginal_indices)

        # Model should be fitted to 2D marginalised samples
        assert isinstance(model, FlowModel)

    def test_single_parameter_marginalisation(self, samples: np.ndarray) -> None:
        """Test marginalisation to a single parameter."""

        marginal_indices = [2]

        with pytest.warns(
            UserWarning, match="Marginalising down to 1D; using KDEModel"
        ):
            model = fit_marginalised_posterior(samples, marginal_indices)

        assert isinstance(model, KDEModel)

    def test_all_parameters(self, samples: np.ndarray) -> None:
        """Test keeping all parameters (no marginalisation)."""
        marginal_indices = [0, 1, 2, 3, 4]

        model = fit_marginalised_posterior(samples, marginal_indices)

        assert model is not None


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


class TestConfigs:
    """Tests for configuration dataclasses."""

    def test_realnvp_config_defaults(self) -> None:
        """Test that RealNVPConfig has correct default values."""
        config = RealNVPConfig()

        assert config.n_scaled_layers == 2
        assert config.n_unscaled_layers == 4
        assert config.learning_rate == 1e-3
        assert config.momentum == 0.9
        assert config.standardize is False
        assert config.temperature == 0.8

    def test_train_config_defaults(self) -> None:
        """Test that TrainConfig has correct default values."""
        config = TrainConfig()

        assert config.batch_size == 64
        assert config.epochs == 10
        assert config.verbose is True

    def test_realnvp_config_frozen(self) -> None:
        """Test that RealNVPConfig is immutable."""
        config = RealNVPConfig()

        with pytest.raises(
            (AttributeError, TypeError)
        ):  # frozen dataclass raises these
            config.learning_rate = 0.01  # type: ignore[misc]

    def test_train_config_frozen(self) -> None:
        """Test that TrainConfig is immutable."""
        config = TrainConfig()

        with pytest.raises(
            (AttributeError, TypeError)
        ):  # frozen dataclass raises these
            config.epochs = 20  # type: ignore[misc]

    def test_realnvp_config_custom_values(self) -> None:
        """Test creating RealNVPConfig with custom values."""
        config = RealNVPConfig(
            n_scaled_layers=3,
            n_unscaled_layers=5,
            learning_rate=1e-4,
            momentum=0.95,
            standardize=True,
            temperature=0.9,
        )

        assert config.n_scaled_layers == 3
        assert config.n_unscaled_layers == 5
        assert config.learning_rate == 1e-4
        assert config.momentum == 0.95
        assert config.standardize is True
        assert config.temperature == 0.9

    def test_train_config_custom_values(self) -> None:
        """Test creating TrainConfig with custom values."""
        config = TrainConfig(batch_size=128, epochs=50, verbose=False)

        assert config.batch_size == 128
        assert config.epochs == 50
        assert config.verbose is False


class TestKDEModel:
    """Tests for the KDEModel class."""

    @pytest.fixture
    def kde(self, rng) -> KDEModel:
        """Sample data for testing."""
        return KDEModel(rng.standard_normal((200, 1)))

    def test_kde_model_initialization(self, kde) -> None:
        """Test that KDEModel initializes without error."""

        assert kde is not None
        assert hasattr(kde, "predict")

    def test_kde_model_predict(self, kde) -> None:
        """Test that KDEModel predict method returns log densities."""

        test_points = np.array([[0.0], [1.0], [-1.0]])
        log_densities = kde.predict(test_points)

        assert log_densities.shape == (3,)
        assert np.all(np.isfinite(log_densities))
