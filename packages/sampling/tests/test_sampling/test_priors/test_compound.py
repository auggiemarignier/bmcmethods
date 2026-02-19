"""Tests for CompoundPrior and PriorComponent."""

import numpy as np
import pytest
from sampling.priors import (
    CompoundPrior,
    GaussianPrior,
    PriorComponent,
    UniformPrior,
)
from sampling.priors._protocols import PriorType


class TestPriorComponent:
    """Tests for PriorComponent dataclass."""

    def test_prior_component_with_list_indices(self) -> None:
        """Test that PriorComponent stores prior function and indices correctly."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = GaussianPrior(mean, covar)
        indices = [0, 1]

        component = PriorComponent(
            type=PriorType.GAUSSIAN, prior_fn=prior_fn, indices=indices
        )

        assert component.prior_fn is prior_fn
        np.testing.assert_array_equal(component.indices, np.array(indices))
        assert component.n == 2

    def test_prior_component_with_slice(self) -> None:
        """Test that PriorComponent can store indices as a slice."""
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        prior_fn = UniformPrior(lower, upper)
        indices = slice(0, 2)

        component = PriorComponent(
            type=PriorType.UNIFORM, prior_fn=prior_fn, indices=indices
        )

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
        gaussian_component = PriorComponent(
            type=PriorType.GAUSSIAN, prior_fn=gaussian_prior, indices=[0, 1]
        )

        # Uniform prior on last two parameters
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        uniform_prior = UniformPrior(lower, upper)
        uniform_component = PriorComponent(
            type=PriorType.UNIFORM, prior_fn=uniform_prior, indices=[2, 3]
        )

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

        # Gaussian prior log-prob at [1.0, -1.0] is -1, uniform prior log-prob within bounds is 0, plus normalisations
        expected_log_prior = -1.0 + sum(
            [
                component.prior_fn._normalisation
                for component in compound_prior.prior_components
            ]
        )
        np.testing.assert_almost_equal(log_prior, expected_log_prior)

    def test_compound_prior_invalid_model(self, compound_prior: CompoundPrior) -> None:
        """Test the compound prior with a model that has out-of-bounds parameters."""
        # Test point outside uniform prior
        model = np.array([0.1, -0.1, 2.0, -0.5])
        log_prior_out_uniform = compound_prior(model)
        assert log_prior_out_uniform == -np.inf

    def test_compound_prior_sample_shape(self, compound_prior: CompoundPrior) -> None:
        rng = np.random.default_rng(42)
        samples = compound_prior.sample(10, rng)
        assert samples.shape == (10, compound_prior.n)

    def test_compound_prior_sample_reproducibility(
        self, compound_prior: CompoundPrior
    ) -> None:
        rng1 = np.random.default_rng(2024)
        rng2 = np.random.default_rng(2024)
        samples1 = compound_prior.sample(5, rng1)
        samples2 = compound_prior.sample(5, rng2)
        np.testing.assert_array_equal(samples1, samples2)

    def test_compound_prior_batched_single_model(
        self, compound_prior: CompoundPrior
    ) -> None:
        """Test that batched evaluation works for a single model."""
        model = np.array([[0.0, 0.0, 0.0, 0.0]])
        log_priors = compound_prior(model)

        assert log_priors.ndim == 0
        assert np.isfinite(log_priors)

    def test_compound_prior_batched_multiple_models(
        self, compound_prior: CompoundPrior
    ) -> None:
        """Test that batched evaluation works for multiple models."""
        models = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5, -0.5],
                [0.5, 0.5, 0.25, 0.75],
            ]
        )
        log_priors = compound_prior(models)

        assert log_priors.shape == (3,)
        assert np.all(np.isfinite(log_priors))

    def test_compound_prior_batched_mixed_valid_invalid(
        self, compound_prior: CompoundPrior
    ) -> None:
        """Test batched evaluation with some models out of bounds."""
        models = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],  # valid
                [0.0, 0.0, 2.0, 0.0],  # out of uniform bounds
                [1.0, 1.0, -0.5, -0.5],  # valid
                [0.0, 0.0, -2.0, 0.0],  # out of uniform bounds
            ]
        )
        log_priors = compound_prior(models)

        assert log_priors.shape == (4,)
        assert np.isfinite(log_priors[0])
        assert log_priors[1] == -np.inf
        assert np.isfinite(log_priors[2])
        assert log_priors[3] == -np.inf

    def test_compound_prior_batched_consistent_with_individual_calls(self) -> None:
        """Test that batched evaluation gives same results as individual calls."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])

        gaussian_prior = GaussianPrior(mean, covar)
        uniform_prior = UniformPrior(lower, upper)
        compound_prior = CompoundPrior(
            [
                PriorComponent(
                    type=PriorType.GAUSSIAN, prior_fn=gaussian_prior, indices=[0, 1]
                ),
                PriorComponent(
                    type=PriorType.UNIFORM, prior_fn=uniform_prior, indices=[2, 3]
                ),
            ],
        )

        models = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, -1.0, 0.5, -0.5]),
            np.array([0.5, 0.5, 0.25, 0.75]),
            np.array([0.1, 0.1, -0.8, 0.8]),
        ]

        # Individual calls
        individual_results = np.array([compound_prior(m) for m in models])

        # Batched call
        batched_models = np.array(models)
        batched_results = compound_prior(batched_models)

        assert batched_results.shape == (4,)
        np.testing.assert_allclose(individual_results, batched_results)

    def test_compound_prior_gradient_single_model(
        self, compound_prior: CompoundPrior
    ) -> None:
        """Gradient for a single model equals sum of component gradients."""
        model = np.array([1.0, -1.0, 0.0, 0.0])
        grad = compound_prior.gradient(model)

        # Gaussian on first two params with identity inv_covar -> gradient = -diff
        expected = np.array([-(1.0 - 0.0), -(-1.0 - 0.0), 0.0, 0.0])
        np.testing.assert_allclose(grad, expected)

    def test_compound_prior_gradient_batched_models(
        self, compound_prior: CompoundPrior
    ) -> None:
        """Batched gradient returns correct shape and matches individual evaluations."""
        models = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, -1.0, 0.5, -0.5],
                [0.5, 0.5, 0.25, 0.75],
            ]
        )

        batched_grad = compound_prior.gradient(models)
        assert batched_grad.shape == (3, compound_prior.n)

        for i in range(models.shape[0]):
            np.testing.assert_allclose(
                batched_grad[i], compound_prior.gradient(models[i])
            )

    def test_compound_prior_gradient_batched_consistent_with_individual_calls(
        self,
    ) -> None:
        """Batched gradient equals stacking of individual gradients for a constructed compound prior."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])

        gaussian_prior = GaussianPrior(mean, covar)
        uniform_prior = UniformPrior(lower, upper)
        compound_prior = CompoundPrior(
            [
                PriorComponent(
                    type=PriorType.GAUSSIAN, prior_fn=gaussian_prior, indices=[0, 1]
                ),
                PriorComponent(
                    type=PriorType.UNIFORM, prior_fn=uniform_prior, indices=[2, 3]
                ),
            ],
        )

        models = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, -1.0, 0.5, -0.5]),
            np.array([0.5, 0.5, 0.25, 0.75]),
            np.array([0.1, 0.1, -0.8, 0.8]),
        ]

        individual_results = np.array([compound_prior.gradient(m) for m in models])
        batched_results = compound_prior.gradient(np.array(models))

        assert batched_results.shape == individual_results.shape
        np.testing.assert_allclose(individual_results, batched_results)

    def test_initialisation_from_dict(self) -> None:
        """Test that CompoundPrior can be initialised from a configuration dictionary."""
        config_dict = {
            "components": [
                {
                    "type": "gaussian",
                    "mean": [0.0, 0.0],
                    "inv_covar": [[1.0, 0.0], [0.0, 1.0]],
                    "indices": [0, 1],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-1.0, -1.0],
                    "upper_bounds": [1.0, 1.0],
                    "indices": [2, 3],
                },
                {
                    "type": "wrapped_uniform",
                    "lower_bounds": [0.0],
                    "upper_bounds": [360.0],
                    "indices": [4],
                },
            ]
        }

        compound_prior = CompoundPrior.from_dict(config_dict)

        assert isinstance(compound_prior, CompoundPrior)
        assert len(compound_prior.prior_components) == 3
        assert compound_prior.prior_components[0].type == PriorType.GAUSSIAN
        assert compound_prior.prior_components[1].type == PriorType.UNIFORM
        assert compound_prior.prior_components[2].type == PriorType.WRAPPED_UNIFORM
