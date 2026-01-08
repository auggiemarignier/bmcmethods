"""Tests for GaussianPrior."""

import numpy as np
import pytest
from sampling.priors import GaussianPrior


def test_gaussian_prior_n() -> None:
    """Test that Gaussian prior has correct number of parameters."""
    mean = np.array([1.0, 2.0, 3.0])
    covar = np.eye(3)
    prior_fn = GaussianPrior(mean, covar)
    assert prior_fn.n == 3


def test_gaussian_config_params_expose_mean_and_covar() -> None:
    """Gaussian prior should expose mean and covariance via config_params in order and by reference."""
    mean = np.array([1.0, -2.0, 3.0])
    covar = np.array(
        [
            [2.0, 0.1, 0.0],
            [0.1, 1.5, 0.2],
            [0.0, 0.2, 3.0],
        ]
    )
    prior = GaussianPrior(mean, covar)
    assert isinstance(prior, GaussianPrior)

    cfg = prior.config_params
    assert isinstance(cfg, list)
    assert len(cfg) == 2

    # Check identity and values
    assert cfg[0] is mean
    assert cfg[1] is covar
    np.testing.assert_array_equal(cfg[0], mean)
    np.testing.assert_array_equal(cfg[1], covar)

    # Shape sanity
    assert cfg[0].shape == mean.shape
    assert cfg[1].shape == covar.shape


def test_gaussian_prior_at_mean() -> None:
    """Test that log-prior is equal to the normalisation at the mean."""
    mean = np.array([1.0, 2.0, 3.0])
    covar = np.eye(3)
    prior_fn = GaussianPrior(mean, covar)

    log_prior = prior_fn(mean)
    assert log_prior == prior_fn._normalisation


def test_gaussian_prior_symmetric() -> None:
    """Test that log-prior is symmetric around the mean."""
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    prior_fn = GaussianPrior(mean, covar)

    delta = np.array([1.0, 1.0])
    log_prior_plus = prior_fn(mean + delta)
    log_prior_minus = prior_fn(mean - delta)

    np.testing.assert_allclose(log_prior_plus, log_prior_minus)


def test_gaussian_prior_decreases_with_distance() -> None:
    """Test that log-prior decreases as we move away from the mean."""
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    prior_fn = GaussianPrior(mean, covar)

    close = np.array([0.1, 0.1])
    far = np.array([1.0, 1.0])

    log_prior_close = prior_fn(close)
    log_prior_far = prior_fn(far)

    assert log_prior_close > log_prior_far


def test_gaussian_prior_with_correlation() -> None:
    """Test Gaussian prior with correlated covariance."""
    mean = np.array([0.0, 0.0])
    covar = np.array([[1.0, 0.5], [0.5, 1.0]])
    prior_fn = GaussianPrior(mean, covar)

    params = np.array([1.0, 1.0])
    log_prior = prior_fn(params)

    # Should be finite and negative
    assert np.isfinite(log_prior)
    assert log_prior < 0


def test_gaussian_prior_invalid_non_symmetric_covariance() -> None:
    """Test that non-symmetric covariance raises ValueError."""
    mean = np.array([0.0, 0.0])
    covar = np.array([[1.0, 0.5], [0.3, 1.0]])  # Non-symmetric

    with pytest.raises(ValueError, match="symmetric"):
        GaussianPrior(mean, covar)


def test_gaussian_prior_invalid_negative_eigenvalue() -> None:
    """Test that covariance with negative eigenvalues raises ValueError."""
    mean = np.array([0.0, 0.0])
    covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

    with pytest.raises(ValueError, match="positive semidefinite"):
        GaussianPrior(mean, covar)


def test_gaussian_prior_invalid_shape_mismatch() -> None:
    """Test that mismatched mean and covariance shapes raise ValueError."""
    mean = np.array([0.0, 0.0])
    covar = np.eye(3)  # Wrong size

    with pytest.raises(ValueError, match="shape"):
        GaussianPrior(mean, covar)


def test_gaussian_prior_diagonal_covariance() -> None:
    """Test Gaussian prior with diagonal covariance (independent parameters)."""
    mean = np.array([1.0, 2.0, 3.0])
    inv_variances = np.array([0.5, 1.0, 2.0])
    inv_covar = np.diag(inv_variances)
    prior_fn = GaussianPrior(mean, inv_covar)

    # Test a point one standard deviation away in first dimension
    params = mean.copy()
    params[0] += np.sqrt(1 / inv_variances[0])
    log_prior = prior_fn(params)

    # Expected: -0.5 * (1^2) + normalisation = -0.5 + normalisation
    np.testing.assert_allclose(log_prior, -0.5 + prior_fn._normalisation, rtol=1e-10)


def test_gaussian_prior_sample_shape() -> None:
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    prior = GaussianPrior(mean, covar)
    rng = np.random.default_rng(42)
    samples = prior.sample(10, rng)
    assert samples.shape == (10, 2)


def test_gaussian_prior_sample_mean() -> None:
    mean = np.array([1.0, -1.0])
    covar = np.eye(2)
    prior = GaussianPrior(mean, covar)
    rng = np.random.default_rng(123)
    samples = prior.sample(1000, rng)
    sample_mean = np.mean(samples, axis=0)
    np.testing.assert_allclose(sample_mean, mean, atol=0.1)


def test_gaussian_prior_sample_reproducibility() -> None:
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    prior = GaussianPrior(mean, covar)
    rng1 = np.random.default_rng(2024)
    rng2 = np.random.default_rng(2024)
    samples1 = prior.sample(5, rng1)
    samples2 = prior.sample(5, rng2)
    np.testing.assert_array_equal(samples1, samples2)
