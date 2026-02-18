"""Tests for priors with wrapped bounds."""

import numpy as np
import pytest
from sampling.priors import UniformPrior, WrappedUniformPrior


def test_valid_initialisation() -> None:
    """Test that the prior is correctly initialised."""
    lower_bounds = np.array([-180.0])
    upper_bounds = np.array([180.0])
    prior = WrappedUniformPrior(lower_bounds, upper_bounds)

    assert issubclass(WrappedUniformPrior, UniformPrior)
    np.testing.assert_array_equal(prior.lower_bounds, np.array([-180.0]))
    np.testing.assert_array_equal(prior.upper_bounds, np.array([180.0]))
    assert prior.n == 1
    np.testing.assert_array_equal(prior.config_params[0], lower_bounds)
    np.testing.assert_array_equal(prior.config_params[1], upper_bounds)


@pytest.fixture
def wrapped_prior() -> WrappedUniformPrior:
    return WrappedUniformPrior(np.array([-180.0]), np.array([180.0]))


@pytest.mark.parametrize(
    "model_param,expected_wrapped",
    [
        (-181.0, 179.0),  # Just below lower bound should wrap to just below upper bound
        (181.0, -179.0),  # Just above upper bound should wrap to just above lower bound
        (180.0, -180.0),  # Exactly at upper bound should wrap to lower bound
        (-180.0, -180.0),  # Exactly at lower bound should stay the same
        (360.0, 0.0),  # Exactly one full range above should wrap to the same point
        (-360.0, 0.0),  # Exactly one full range below should wrap to the same point
        (540.0, -180.0),  # One and a half ranges above should wrap to lower bound
        (0.0, 0.0),  # Within bounds should not change
    ],
)
def test_model_parameter_wrapping(
    wrapped_prior: WrappedUniformPrior, model_param: float, expected_wrapped: float
) -> None:
    """Test that model parameters are correctly wrapped around the specified bounds."""

    model_params = np.array([model_param])
    wrapped_model_params = wrapped_prior._wrap(model_params)
    expected_wrapped_model_params = np.array([expected_wrapped])

    np.testing.assert_allclose(wrapped_model_params, expected_wrapped_model_params)
    assert wrapped_model_params.shape == model_params.shape


def test_model_parameter_wrapping_batch(wrapped_prior: WrappedUniformPrior) -> None:
    """Test that model parameters are correctly wrapped for a batch of parameters."""

    model_params = np.array([[-181.0], [180.0], [360.0], [-360.0], [540.0], [0.0]])
    expected_wrapped_params = np.array(
        [[179.0], [-180.0], [0.0], [0.0], [-180.0], [0.0]]
    )

    wrapped_model_params = wrapped_prior._wrap(model_params)

    np.testing.assert_allclose(wrapped_model_params, expected_wrapped_params)
    assert wrapped_model_params.shape == model_params.shape


def test_multidimensional_batched_wrapping() -> None:
    """Test that multidimensional model parameters are correctly wrapped for a batch of parameters."""

    lower = np.array([-180.0, -90.0])
    upper = np.array([180.0, 90.0])
    wrapped_prior = WrappedUniformPrior(lower_bounds=lower, upper_bounds=upper)

    model_params = np.array(
        [[-181.0, -91.0], [180.0, 90.0], [360.0, 180.0], [-360.0, -180.0]]
    )
    expected_wrapped_params = np.array(
        [[179.0, 89.0], [-180.0, -90.0], [0.0, 0.0], [0.0, 0.0]]
    )

    wrapped_model_params = wrapped_prior._wrap(model_params)

    np.testing.assert_allclose(wrapped_model_params, expected_wrapped_params)
    assert wrapped_model_params.shape == model_params.shape


@pytest.mark.parametrize(
    "model_param, expected_param",
    [
        (-181.0, 179.0),   # out of bounds, should wrap
        (-180.0, -180.0),  # lower boundary
        (0.0, 0.0),        # within bounds
        (179.0, 179.0),    # upper boundary - 1
        (180.0, 180.0),    # upper boundary
    ],
)
def test_call(
    wrapped_prior: WrappedUniformPrior, model_param: float, expected_param: float
) -> None:
    """Test that calling the WrappedPrior returns the same log-prior as the base UniformPrior, with wrapping only for out-of-bounds parameters."""

    model_params = np.array([model_param])
    log_prior = wrapped_prior(model_params)

    base_params = np.array([expected_param])
    uniform = UniformPrior(wrapped_prior.lower_bounds, wrapped_prior.upper_bounds)
    expected_log_prior = uniform(base_params)

    np.testing.assert_allclose(log_prior, expected_log_prior)


def test_sample(wrapped_prior: WrappedUniformPrior) -> None:
    """Test that we can sample from the WrappedPrior."""

    rng = np.random.default_rng(seed=42)
    num_samples = 1000
    samples = wrapped_prior.sample(num_samples=num_samples, rng=rng)

    assert samples.shape == (num_samples, wrapped_prior.n)
    # Verify that all samples lie within the wrapped prior bounds
    assert np.all(samples >= wrapped_prior.lower_bounds)
    assert np.all(samples <= wrapped_prior.upper_bounds)
