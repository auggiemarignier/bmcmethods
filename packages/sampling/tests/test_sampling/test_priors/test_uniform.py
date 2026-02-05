"""Tests for UniformPrior."""

from collections.abc import Callable

import numpy as np
import pytest
from sampling.priors import UniformPrior


@pytest.fixture
def lower() -> np.ndarray:
    """Create lower bounds for testing."""
    return np.array([0.0, 0.0])


@pytest.fixture
def upper() -> np.ndarray:
    """Create upper bounds for testing."""
    return np.array([1.0, 1.0])


@pytest.fixture
def valid_uniform_prior(lower: np.ndarray, upper: np.ndarray) -> UniformPrior:
    """Create a valid uniform prior function for testing."""
    return UniformPrior(lower, upper)


@pytest.fixture
def vectorised_uniform_prior(lower: np.ndarray, upper: np.ndarray) -> UniformPrior:
    """Create a valid vectorised uniform prior function for testing."""
    return UniformPrior(lower, upper)


def test_uniform_prior_n(valid_uniform_prior: UniformPrior) -> None:
    """Test that uniform prior has correct number of parameters."""
    assert valid_uniform_prior.n == 2


def test_uniform_config_params_expose_bounds(
    valid_uniform_prior: UniformPrior,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    """Uniform prior should expose lower and upper bounds via config_params in order and by reference."""
    assert isinstance(valid_uniform_prior, UniformPrior)

    cfg = valid_uniform_prior.config_params
    assert isinstance(cfg, list)
    assert len(cfg) == 2

    # Check identity and values
    assert cfg[0] is lower
    assert cfg[1] is upper
    np.testing.assert_array_equal(cfg[0], lower)
    np.testing.assert_array_equal(cfg[1], upper)

    # Shape sanity
    assert cfg[0].shape == lower.shape
    assert cfg[1].shape == upper.shape


def test_uniform_prior_in_bounds(
    valid_uniform_prior: Callable[[np.ndarray], float],
) -> None:
    """Test that log-prior is zero for parameters within bounds."""

    params = np.array([0.5, 0.5])
    log_prior = valid_uniform_prior(params)

    assert log_prior == 0.0


def test_uniform_prior_at_lower_boundary(
    valid_uniform_prior: Callable[[np.ndarray], float], lower: np.ndarray
) -> None:
    """Test that log-prior is zero at the lower boundaries."""
    log_prior_lower = valid_uniform_prior(lower)
    assert log_prior_lower == 0.0


def test_uniform_prior_at_upper_boundary(
    valid_uniform_prior: Callable[[np.ndarray], float], upper: np.ndarray
) -> None:
    """Test that log-prior is zero at the upper boundaries."""
    log_prior_upper = valid_uniform_prior(upper)
    assert log_prior_upper == 0.0


def test_uniform_prior_out_of_bounds_below(
    valid_uniform_prior: Callable[[np.ndarray], float],
) -> None:
    """Test that log-prior is -inf for parameters below lower bound."""
    params = np.array([-0.1, 0.5])
    log_prior = valid_uniform_prior(params)

    assert log_prior == -np.inf


def test_uniform_prior_out_of_bounds_above(
    valid_uniform_prior: Callable[[np.ndarray], float],
) -> None:
    """Test that log-prior is -inf for parameters above upper bound."""
    params = np.array([0.5, 1.1])
    log_prior = valid_uniform_prior(params)

    assert log_prior == -np.inf


def test_uniform_prior_multiple_out_of_bounds(
    valid_uniform_prior: Callable[[np.ndarray], float],
) -> None:
    """Test that log-prior is -inf when multiple parameters are out of bounds."""
    params = np.array([-0.1, 1.1])
    log_prior = valid_uniform_prior(params)

    assert log_prior == -np.inf


def test_uniform_prior_invalid_bounds() -> None:
    """Test that invalid bounds (lower >= upper) raise ValueError."""
    lower = np.array([0.0, 1.0])
    upper = np.array([1.0, 0.5])  # Second bound is invalid

    with pytest.raises(ValueError, match="lower bound must be less than"):
        UniformPrior(lower, upper)


def test_uniform_prior_equal_bounds() -> None:
    """Test that equal bounds raise ValueError."""
    lower = np.array([0.0, 1.0])
    upper = np.array([1.0, 1.0])  # Second bound is equal

    with pytest.raises(ValueError, match="lower bound must be less than"):
        UniformPrior(lower, upper)


def test_uniform_prior_different_ranges() -> None:
    """Test uniform prior with different ranges per dimension."""
    lower = np.array([0.0, -10.0, 5.0])
    upper = np.array([1.0, 10.0, 15.0])
    prior_fn = UniformPrior(lower, upper)

    # In bounds
    params_in = np.array([0.5, 0.0, 10.0])
    assert prior_fn(params_in) == prior_fn._normalisation

    # Out of bounds in third dimension
    params_out = np.array([0.5, 0.0, 20.0])
    assert prior_fn(params_out) == -np.inf


def test_uniform_prior_high_dimensional() -> None:
    """Test uniform prior in high-dimensional space."""
    n_dims = 10
    lower = np.zeros(n_dims)
    upper = np.ones(n_dims)
    prior_fn = UniformPrior(lower, upper)

    # All in bounds
    params_in = np.full(n_dims, 0.5)
    assert prior_fn(params_in) == 0.0

    # One out of bounds
    params_out = np.full(n_dims, 0.5)
    params_out[5] = 1.1
    assert prior_fn(params_out) == -np.inf


def test_uniform_prior_sample_shape(valid_uniform_prior: UniformPrior) -> None:
    rng = np.random.default_rng(42)
    samples = valid_uniform_prior.sample(10, rng)
    assert samples.shape == (10, valid_uniform_prior.n)


def test_uniform_prior_sample_bounds(valid_uniform_prior: UniformPrior) -> None:
    rng = np.random.default_rng(123)
    samples = valid_uniform_prior.sample(100, rng)
    assert np.all(samples >= valid_uniform_prior.lower_bounds)
    assert np.all(samples <= valid_uniform_prior.upper_bounds)


def test_uniform_prior_sample_reproducibility(
    valid_uniform_prior: UniformPrior,
) -> None:
    rng1 = np.random.default_rng(2024)
    rng2 = np.random.default_rng(2024)
    samples1 = valid_uniform_prior.sample(5, rng1)
    samples2 = valid_uniform_prior.sample(5, rng2)
    np.testing.assert_array_equal(samples1, samples2)


def test_uniform_prior_vectorised_single_model(
    vectorised_uniform_prior: UniformPrior,
) -> None:
    """Test that vectorised evaluation works for a single model."""
    params = np.array([[0.5, 0.5]])
    log_priors = vectorised_uniform_prior(params)

    assert log_priors.ndim == 0
    assert log_priors == vectorised_uniform_prior._normalisation


def test_uniform_prior_vectorised_multiple_models(
    vectorised_uniform_prior: UniformPrior,
) -> None:
    """Test that vectorised evaluation works for multiple models."""
    params = np.array(
        [
            [0.5, 0.5],
            [0.25, 0.75],
            [0.0, 1.0],
        ]
    )
    log_priors = vectorised_uniform_prior(params)

    assert log_priors.shape == (3,)
    np.testing.assert_array_equal(
        log_priors,
        [vectorised_uniform_prior._normalisation] * 3,
    )


def test_uniform_prior_vectorised_mixed_bounds(
    vectorised_uniform_prior: UniformPrior,
) -> None:
    """Test vectorised evaluation with some models in and out of bounds."""
    params = np.array(
        [
            [0.5, 0.5],  # in bounds
            [-0.1, 0.5],  # out of bounds (below)
            [0.5, 1.1],  # out of bounds (above)
            [0.0, 0.0],  # at lower boundary
            [1.0, 1.0],  # at upper boundary
        ]
    )
    log_priors = vectorised_uniform_prior(params)

    assert log_priors.shape == (5,)
    assert log_priors[0] == vectorised_uniform_prior._normalisation
    assert log_priors[1] == -np.inf
    assert log_priors[2] == -np.inf
    assert log_priors[3] == vectorised_uniform_prior._normalisation
    assert log_priors[4] == vectorised_uniform_prior._normalisation


def test_uniform_prior_vectorised_consistent_with_individual_calls(
    lower: np.ndarray, upper: np.ndarray
) -> None:
    """Test that batched evaluation gives same results as individual calls."""
    scalar_prior = UniformPrior(lower, upper)
    vectorised_prior = UniformPrior(lower, upper)

    models = [
        np.array([0.5, 0.5]),
        np.array([0.25, 0.75]),
        np.array([0.0, 1.0]),
        np.array([0.1, 0.9]),
    ]

    # Individual calls
    individual_results = np.array([scalar_prior(m) for m in models])

    # Batched call
    batched_models = np.array(models)
    batched_results = vectorised_prior(batched_models)

    assert batched_results.shape == (4,)
    np.testing.assert_array_equal(individual_results, batched_results)
