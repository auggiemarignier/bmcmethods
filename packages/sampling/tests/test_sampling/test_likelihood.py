"""Test the likelihood functions."""

import pickle

import numpy as np
import pytest
from sampling.likelihood.gaussian import (
    GaussianLikelihood,
    GaussianLikelihoodState,
    _validate_covariance_matrix,
    gaussian_log_likelihood,
    grad_gaussian_loglikelihood,
)


def _dummy_forward_fn(model_params: np.ndarray) -> np.ndarray:
    """A simple forward function for testing purposes."""
    return model_params * 2.0


def _dummy_forward_fn_gradient(model_params: np.ndarray) -> np.ndarray:
    """A simple forward function gradient for testing purposes."""
    if model_params.ndim == 1:
        return 2 * np.eye(model_params.size)
    else:
        return np.stack(
            [2 * np.eye(model_params.shape[1]) for _ in range(model_params.shape[0])],
            axis=0,
        )


@pytest.fixture(
    params=[
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0]),
    ],
    ids=["full_covariance", "diagonal_covariance", "scalar_covariance"],
)
def state(request) -> GaussianLikelihoodState:
    observed_data = np.array([1.0, 2.0, 3.0])
    return GaussianLikelihoodState(
        forward_fn=_dummy_forward_fn,
        observed_data=observed_data,
        inv_covar=request.param,
        forward_fn_gradient=_dummy_forward_fn_gradient,
    )


def test_gaussian_log_likelihood_perfect_match(state: GaussianLikelihoodState) -> None:
    """Test the Gaussian log likelihood function."""
    model_params = (
        state.observed_data / 2.0
    )  # So that predicted data matches observed data
    log_likelihood = gaussian_log_likelihood(model_params, state)

    expected_log_likelihood = 0.0  # Perfect match
    assert np.isclose(log_likelihood, expected_log_likelihood)


def test_gaussian_log_likelihood_non_perfect_match(
    state: GaussianLikelihoodState,
) -> None:
    """Test the Gaussian log likelihood function."""
    model_params = state.observed_data
    log_likelihood = gaussian_log_likelihood(model_params, state)

    expected_log_likelihood = -0.5 * np.sum(
        state.observed_data**2
    )  # For this particular forward function
    assert np.isclose(log_likelihood, expected_log_likelihood)


def test_grad_gaussian_loglikelihood(state: GaussianLikelihoodState) -> None:
    """Test the Gaussian loglikelihood gradient function."""
    model_params = state.observed_data / 2.0
    gradient = grad_gaussian_loglikelihood(model_params, state)

    expected_gradient = np.array(
        [0.0, 0.0, 0.0]
    )  # Gradient should be zero at the maximum likelihood point
    np.testing.assert_allclose(gradient, expected_gradient)

    model_params = state.observed_data
    gradient = grad_gaussian_loglikelihood(model_params, state)
    expected_gradient = 2 * (state.observed_data - 2 * model_params)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_gaussian_likelihood_with_gradient_diagonal_covariance() -> None:
    """Test Gaussian likelihood gradient with a diagonal inverse covariance."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([2.0, 1.5, 1.0])

    likelihood_fn = GaussianLikelihood(
        _dummy_forward_fn,
        observed_data,
        covar,
        forward_fn_gradient=_dummy_forward_fn_gradient,
    )

    # At the maximum-likelihood point, the gradient should be zero.
    model_params = observed_data / 2.0
    gradient = likelihood_fn.gradient(model_params)
    expected_gradient = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(gradient, expected_gradient)

    # Away from the optimum, the gradient should include the diagonal scaling.
    model_params = observed_data
    gradient = likelihood_fn.gradient(model_params)
    expected_gradient = 2 * covar * (observed_data - 2 * model_params)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_gaussian_likelihood_with_gradient_scalar_covariance() -> None:
    """Test Gaussian likelihood gradient with a scalar inverse covariance."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([1.5])

    likelihood_fn = GaussianLikelihood(
        _dummy_forward_fn,
        observed_data,
        covar,
        forward_fn_gradient=_dummy_forward_fn_gradient,
    )

    # At the maximum-likelihood point, the gradient should be zero.
    model_params = observed_data / 2.0
    gradient = likelihood_fn.gradient(model_params)
    expected_gradient = np.array([0.0, 0.0, 0.0])
    np.testing.assert_allclose(gradient, expected_gradient)

    # Away from the optimum, the gradient should be uniformly scaled by the scalar.
    model_params = observed_data
    gradient = likelihood_fn.gradient(model_params)
    expected_gradient = 2 * covar[0] * (observed_data - 2 * model_params)
    np.testing.assert_allclose(gradient, expected_gradient)


def test_gaussian_likelihood_gradient_called_without_forward_fn_gradient() -> None:
    """Test that calling the gradient method without a forward_fn_gradient raises an error."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    likelihood_fn = GaussianLikelihood(_dummy_forward_fn, observed_data, covar)

    model_params = observed_data / 2.0

    with pytest.raises(
        RuntimeError, match="Gradient function for the forward model must be provided"
    ):
        likelihood_fn.gradient(model_params)


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
        "sampling.likelihood.gaussian._validate_covariance_matrix",
        fake_validate_covariance_matrix,
    )

    # Should not raise, and should not call the fake validator
    _ = GaussianLikelihood(
        lambda x: observed_data, observed_data, inv_covar, validate_covariance=False
    )
    assert not called


def test_validate_covariance_matrix_scalar_positive():
    # Should not raise
    _validate_covariance_matrix(np.array([1.0]), 1)


def test_validate_covariance_matrix_scalar_nonpositive():
    with pytest.raises(ValueError, match="Variance scalar must be positive"):
        _validate_covariance_matrix(np.array([0.0]), 1)
    with pytest.raises(ValueError, match="Variance scalar must be positive"):
        _validate_covariance_matrix(np.array([-1.0]), 1)


def test_validate_covariance_matrix_diagonal_positive():
    # Should not raise
    _validate_covariance_matrix(np.array([1.0, 2.0, 3.0]), 3)


def test_validate_covariance_matrix_diagonal_nonpositive():
    with pytest.raises(
        ValueError, match="Covariance diagonal elements must be positive"
    ):
        _validate_covariance_matrix(np.array([1.0, 0.0, 2.0]), 3)
    with pytest.raises(
        ValueError, match="Covariance diagonal elements must be positive"
    ):
        _validate_covariance_matrix(np.array([1.0, -2.0, 2.0]), 3)


def test_validate_covariance_matrix_full_valid():
    mat = np.eye(2)
    _validate_covariance_matrix(mat, 2)


def test_validate_covariance_matrix_full_wrong_shape():
    mat = np.eye(3)
    with pytest.raises(
        ValueError, match="Covariance matrix must be of shape \\(2, 2\\)"
    ):
        _validate_covariance_matrix(mat, 2)


def test_validate_covariance_matrix_full_nonsymmetric():
    mat = np.array([[1.0, 2.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="Covariance matrix must be symmetric"):
        _validate_covariance_matrix(mat, 2)


def test_validate_covariance_matrix_full_not_pos_semidefinite():
    mat = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(
        ValueError, match="Inverse covariance matrix must be positive semidefinite"
    ):
        _validate_covariance_matrix(mat, 2)


class TestBatchedVsScalar:
    """Test that batched and scalar evaluations produce identical results."""

    observed_data = np.array([1.0, 2.0, 3.0])
    model_params_single = np.array([0.5, 1.0, 1.5])
    model_params_batch = np.array(
        [
            [0.5, 1.0, 1.5],
            [0.4, 0.9, 1.4],
            [0.6, 1.1, 1.6],
        ]
    )

    @pytest.mark.parametrize(
        "inv_covar",
        [
            np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]),
            np.array([2.0, 1.5, 1.0]),
            np.array([1.5]),
        ],
        ids=["full_covariance", "diagonal_covariance", "scalar_covariance"],
    )
    def test_scalar_vs_batched_batch(self, inv_covar):
        """Test scalar and batched evaluation with batch of model parameters."""
        likelihood = GaussianLikelihood(
            _dummy_forward_fn,
            self.observed_data,
            inv_covar,
        )

        # Test batch evaluation
        result_batch = likelihood(self.model_params_batch)
        assert result_batch.shape == (3,)

        # Test that batch includes the single result
        result_individual = np.array(
            [likelihood(params) for params in self.model_params_batch]
        )
        np.testing.assert_allclose(result_individual, result_batch)

    def test_scalar_return_type(self):
        """Test that scalar mode returns a 0D NumPy array (scalar array), not a Python float."""
        inv_covar = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])

        likelihood = GaussianLikelihood(
            _dummy_forward_fn,
            self.observed_data,
            inv_covar,
        )

        result = likelihood(self.model_params_single)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 0  # scalar array

    def test_batched_return_type(self):
        """Test that batched mode returns array for batch input."""
        inv_covar = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])

        likelihood = GaussianLikelihood(
            _dummy_forward_fn,
            self.observed_data,
            inv_covar,
        )

        result = likelihood(self.model_params_batch)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
