"""Test the likelihood functions."""

import pickle

import numpy as np
import pytest
from sampling.likelihood import GaussianLikelihood


def _dummy_forward_fn(model_params: np.ndarray) -> np.ndarray:
    """A simple forward function for testing purposes."""
    return model_params * 2.0


def test_gaussian_likelihood_factory() -> None:
    """Test the Gaussian likelihood factory."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    likelihood_fn = GaussianLikelihood(_dummy_forward_fn, observed_data, covar)

    model_params = observed_data / 2.0  # So that predicted data matches observed data
    log_likelihood = likelihood_fn(model_params)

    expected_log_likelihood = 0.0  # Perfect match

    assert np.isclose(log_likelihood, expected_log_likelihood)


def test_invalid_asymmetric_covariance_matrix() -> None:
    """Test that an asymmetrical covariance matrix raises a ValueError."""
    observed_data = np.array([1.0, 2.0])
    covar = np.array([[1.0, 2.0], [0.0, 1.0]])  # Asymmetric

    with pytest.raises(ValueError, match="Covariance matrix must be symmetric."):
        GaussianLikelihood(_dummy_forward_fn, observed_data, covar)


def test_invalid_non_positive_semidefinite_covariance_matrix() -> None:
    """Test that a non-positive semidefinite covariance matrix raises a ValueError."""
    observed_data = np.array([1.0, 2.0])
    covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

    with pytest.raises(
        ValueError, match="Inverse covariance matrix must be positive semidefinite."
    ):
        GaussianLikelihood(_dummy_forward_fn, observed_data, covar)


def test_invalid_data_vector_dimension() -> None:
    """Test that a non-one-dimensional data vector raises a ValueError."""
    observed_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array
    covar = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="Data vector must be one-dimensional."):
        GaussianLikelihood(_dummy_forward_fn, observed_data, covar)


def test_invalid_forward_function_output_dimension() -> None:
    """Test that a forward function returning incorrect output dimension raises a ValueError.

    This only happens when an example_model is provided to the factory.
    """
    observed_data = np.array([1.0, 2.0])
    covar = np.array([[1.0, 0.0], [0.0, 1.0]])

    def bad_forward_fn(model_params: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])  # wrong length

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        GaussianLikelihood(
            bad_forward_fn, observed_data, covar, example_model=np.array([0.0, 0.0])
        )

    try:
        _ = GaussianLikelihood(
            bad_forward_fn, observed_data, covar
        )  # a bad forward function but no example_model, so no check
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    try:
        _ = GaussianLikelihood(
            _dummy_forward_fn, observed_data, covar, example_model=np.array([0.0, 0.0])
        )  # valid forward function verification
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_invalid_covariance_matrix_size() -> None:
    """Test that a covariance matrix with incorrect size raises a ValueError."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 instead of 3x3

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        GaussianLikelihood(_dummy_forward_fn, observed_data, covar)


def test_gaussian_likelihood_picklable():
    """Test that the GaussianLikelihood object is picklable."""
    observed_data = np.array([1.0, 2.0])
    inv_covar = np.eye(2)
    likelihood = GaussianLikelihood(_dummy_forward_fn, observed_data, inv_covar)
    pickled = pickle.dumps(likelihood)
    unpickled = pickle.loads(pickled)
    params = np.array([0.0, 0.0])
    assert np.isclose(likelihood(params), unpickled(params))


def test_gaussian_likelihood_no_covariance_validation(monkeypatch):
    """Test that _validate_covariance_matrix is not called when validate_covariance is False."""
    observed_data = np.array([1.0, 2.0])
    inv_covar = np.array([[1.0, 0.0], [0.0, 1.0]])
    called = False

    def fake_validate_covariance_matrix(covar, N):
        nonlocal called
        called = True
        raise AssertionError("Should not be called!")

    monkeypatch.setattr(
        "sampling.likelihood._validate_covariance_matrix",
        fake_validate_covariance_matrix,
    )

    # Should not raise, and should not call the fake validator
    _ = GaussianLikelihood(
        lambda x: observed_data, observed_data, inv_covar, validate_covariance=False
    )
    assert not called
