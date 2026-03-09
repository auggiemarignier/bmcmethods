"""Test the posterior module."""

import pickle

import numpy as np
import pytest
from sampling.posterior import Posterior
from sddr.marginalisation import marginalise_samples


def _dummy_likelihood_fn(params):
    """Dummy likelihood function for pickling test."""
    return -0.5 * np.sum(params**2, axis=-1)


def _dummy_prior_fn(params):
    """Dummy prior function for pickling test."""
    return -np.sum(np.abs(params), axis=-1)


def test_posterior():
    """Test the posterior on a simple example."""

    posterior_fn = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)

    params = np.array([1.0, 2.0, -1.5])
    log_posterior = posterior_fn(params)

    expected_log_likelihood = -0.5 * (1.0**2 + 2.0**2 + (-1.5) ** 2)
    expected_log_prior = -(np.abs(1.0) + np.abs(2.0) + np.abs(-1.5))
    expected_log_posterior = expected_log_likelihood + expected_log_prior

    assert np.isclose(log_posterior, expected_log_posterior)


def test_posterior_with_gradient():
    """Test the posterior with gradient functions."""

    def likelihood_gradient_fn(params):
        return -params

    def prior_gradient_fn(params):
        return -np.sign(params)

    posterior_fn = Posterior(
        _dummy_likelihood_fn, _dummy_prior_fn, likelihood_gradient_fn, prior_gradient_fn
    )

    params = np.array([1.0, 2.0, -1.5])
    log_posterior = posterior_fn(params)
    gradient = posterior_fn.gradient(params)

    expected_log_likelihood = -0.5 * (1.0**2 + 2.0**2 + (-1.5) ** 2)
    expected_log_prior = -(np.abs(1.0) + np.abs(2.0) + np.abs(-1.5))
    expected_log_posterior = expected_log_likelihood + expected_log_prior
    expected_gradient = -params - np.sign(params)

    assert np.isclose(log_posterior, expected_log_posterior)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_posterior_gradient_not_available():
    """Test that calling the gradient method raises an error if gradient functions are not provided."""

    posterior_fn = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)

    params = np.array([1.0, 2.0, -1.5])
    with pytest.raises(
        ValueError, match="Gradient functions for likelihood and prior must be provided"
    ):
        posterior_fn.gradient(params)


def test_posterior_picklable():
    """Test that Posterior is picklable and works after unpickling."""
    posterior = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)
    pickled = pickle.dumps(posterior)
    unpickled = pickle.loads(pickled)
    params = np.array([0.5, -0.5])
    assert np.isclose(posterior(params), unpickled(params))


def test_posterior_batched():
    """Test the posterior with batched inputs."""

    posterior_fn = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)

    # Batched parameters: shape (3, 3) - 3 models with 3 parameters each
    params = np.array(
        [
            [1.0, 2.0, -1.5],
            [0.5, -0.5, 1.0],
            [-1.0, 1.5, 0.0],
        ]
    )
    log_posteriors = posterior_fn(params)

    # Expected results for each model
    expected_log_likelihoods = -0.5 * np.sum(params**2, axis=1)
    expected_log_priors = -np.sum(np.abs(params), axis=1)
    expected_log_posteriors = expected_log_likelihoods + expected_log_priors

    assert log_posteriors.shape == (3,)
    np.testing.assert_allclose(log_posteriors, expected_log_posteriors)


def test_posterior_batched_consistent_with_single():
    """Test that batched evaluation gives same results as individual calls."""

    posterior_fn = Posterior(_dummy_likelihood_fn, _dummy_prior_fn)

    models = [
        np.array([1.0, 2.0, -1.5]),
        np.array([0.5, -0.5, 1.0]),
        np.array([-1.0, 1.5, 0.0]),
    ]

    # Individual calls
    individual_results = np.array([posterior_fn(m) for m in models])

    # Batched call
    batched_models = np.array(models)
    batched_results = posterior_fn(batched_models)

    assert batched_results.shape == (3,)
    np.testing.assert_allclose(individual_results, batched_results)


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


def test_early_escape_in_posterior():
    """Test that the posterior returns -inf if the prior is -inf, without evaluating the likelihood."""

    def infinite_prior(params):
        return -np.inf

    def likelihood(params):
        raise RuntimeError("Likelihood should not be evaluated when prior is -inf")

    posterior_fn = Posterior(likelihood, infinite_prior)

    params = np.array([1.0, 2.0, -1.5])
    log_posterior = posterior_fn(params)

    assert log_posterior == -np.inf


def test_early_escape_in_posterior_batched():
    """Test batched early escape: -inf prior entries yield -inf posterior without evaluating their likelihood."""

    def batched_infinite_prior(params: np.ndarray) -> np.ndarray:
        """Return -inf for models whose first parameter is negative, finite otherwise."""
        params = np.asarray(params)
        # Base finite prior (matching the dummy prior style).
        prior = -np.sum(np.abs(params), axis=-1, dtype=float)
        # Mark as -inf where first coordinate is negative.
        mask = params[..., 0] < 0
        prior[mask] = -np.inf
        return prior

    def likelihood(params: np.ndarray) -> np.ndarray:
        """Likelihood must not be evaluated for models with negative first coordinate."""
        params = np.asarray(params)
        # In batched mode, Posterior may call the likelihood on a subset of models.
        # Ensure we never see models that should have -inf prior (first coord < 0).
        assert not np.any(params[..., 0] < 0), (
            "Likelihood should not be evaluated for models with -inf prior"
        )
        return -0.5 * np.sum(params**2, axis=-1)

    posterior_fn = Posterior(likelihood, batched_infinite_prior)

    models = np.array(
        [
            [1.0, 2.0, -1.5],   # finite prior
            [-1.0, 0.0, 0.5],   # -inf prior (first coordinate negative)
            [0.5, -0.5, 1.0],   # finite prior
        ]
    )

    log_posteriors = posterior_fn(models)

    # Should return one posterior value per model in the batch.
    assert log_posteriors.shape == (3,)

    # Second model has -inf prior -> posterior must be -inf.
    assert log_posteriors[1] == -np.inf

    # Other entries must be finite.
    assert np.isfinite(log_posteriors[0])
    assert np.isfinite(log_posteriors[2])
