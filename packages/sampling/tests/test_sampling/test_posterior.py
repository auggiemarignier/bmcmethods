"""Test the posterior module."""

import pickle

import numpy as np
import pytest
from sampling.posterior import Posterior
from sddr.marginalisation import marginalise_samples


def _dummy_likelihood_fn(params):
    """Dummy likelihood function for pickling test."""
    return -0.5 * np.sum(params**2)


def _dummy_prior_fn(params):
    """Dummy prior function for pickling test."""
    return -np.sum(np.abs(params))


def test_posterior():
    """Test the posterior on a simple example."""

    posterior_fn = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)

    params = np.array([1.0, 2.0, -1.5])
    log_posterior = posterior_fn(params)

    expected_log_likelihood = -0.5 * (1.0**2 + 2.0**2 + (-1.5) ** 2)
    expected_log_prior = -(np.abs(1.0) + np.abs(2.0) + np.abs(-1.5))
    expected_log_posterior = expected_log_likelihood + expected_log_prior

    assert np.isclose(log_posterior, expected_log_posterior)


def test_posterior_picklable():
    """Test that Posterior is picklable and works after unpickling."""
    posterior = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)
    pickled = pickle.dumps(posterior)
    unpickled = pickle.loads(pickled)
    params = np.array([0.5, -0.5])
    assert np.isclose(posterior(params), unpickled(params))


@pytest.fixture
def samples() -> np.ndarray:
    """Fixture for sample posterior samples."""
    return np.array(
        [
            [0.5, 1.0, -0.5, 2.0],
            [1.5, -1.0, 0.5, -2.0],
            [-0.5, 0.0, 1.5, 3.0],
        ]
    )


def test_marginalise_posterior_samples_with_indices(samples):
    """Test the marginalisation of posterior samples using indices to select which parameters we want to keep."""

    marginal_samples = marginalise_samples(samples, [0, 1])

    expected_marginal_samples = np.array(
        [
            [0.5, 1.0],
            [1.5, -1.0],
            [-0.5, 0.0],
        ]
    )

    assert np.array_equal(marginal_samples, expected_marginal_samples)


def test_marginalise_posterior_samples_with_slice(samples):
    """Test the marginalisation of posterior samples using a slice to select which parameters we want to keep."""

    marginal_samples = marginalise_samples(samples, slice(1, 4))

    expected_marginal_samples = np.array(
        [
            [1.0, -0.5, 2.0],
            [-1.0, 0.5, -2.0],
            [0.0, 1.5, 3.0],
        ]
    )

    assert np.array_equal(marginal_samples, expected_marginal_samples)
