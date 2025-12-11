"""Tests for CompoundPrior and PriorComponent."""

import numpy as np
import pytest
from sddr.priors import (
    CompoundPrior,
    GaussianPrior,
    PriorComponent,
    UniformPrior,
)


class TestPriorComponent:
    """Tests for PriorComponent dataclass."""

    def test_prior_component_with_list_indices(self) -> None:
        """Test that PriorComponent stores prior function and indices correctly."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = GaussianPrior(mean, covar)
        indices = [0, 1]

        component = PriorComponent(prior_fn=prior_fn, indices=indices)

        assert component.prior_fn is prior_fn
        np.testing.assert_array_equal(component.indices, np.array(indices))
        assert component.n == 2

    def test_prior_component_with_slice(self) -> None:
        """Test that PriorComponent can store indices as a slice."""
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        prior_fn = UniformPrior(lower, upper)
        indices = slice(0, 2)

        component = PriorComponent(prior_fn=prior_fn, indices=indices)

        assert component.prior_fn is prior_fn
        np.testing.assert_array_equal(component.indices, np.arange(0, 2))
        assert component.n == 2


class TestCompoundPrior:
    """Tests for compound prior functions combining Gaussian and Uniform priors."""

    @pytest.fixture
    def compound_prior(self) -> CompoundPrior:
        """Create a compound prior for testing."""
        # Gaussian prior on first two parameters
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        gaussian_prior = GaussianPrior(mean, covar)
        gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

        # Uniform prior on last two parameters
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        uniform_prior = UniformPrior(lower, upper)
        uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

        # Combine into compound prior
        return CompoundPrior([gaussian_component, uniform_component])

    def test_compound_prior_n(self, compound_prior: CompoundPrior) -> None:
        """Test that compound prior infers correct number of parameters from components."""
        assert compound_prior.n == 4

    def test_compound_prior_valid_model(self, compound_prior: CompoundPrior) -> None:
        """Test the compound prior with a valid model."""

        # Test point within both priors
        model = np.array(
            [1.0, -1.0, 0.0, 0.0]
        )  # 1 stddev away from the mean of the gaussian prior and within uniform prior
        log_prior = compound_prior(model)

        # Gaussian prior log-prob at [1.0, -1.0] is -1, uniform prior log-prob within bounds is 0
        expected_log_prior = -1.0
        np.testing.assert_almost_equal(log_prior, expected_log_prior)

    def test_compound_prior_invalid_model(self, compound_prior: CompoundPrior) -> None:
        """Test the compound prior with a model that has out-of-bounds parameters."""
        # Test point outside uniform prior
        model = np.array([0.1, -0.1, 2.0, -0.5])
        log_prior_out_uniform = compound_prior(model)
        assert log_prior_out_uniform == -np.inf
