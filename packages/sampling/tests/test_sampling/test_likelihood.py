"""Test the likelihood functions."""

import pickle

import numpy as np
import pytest
from sampling.likelihood import GaussianLikelihood, _validate_covariance_matrix


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


class TestExponentialTermFunctions:
    class Dummy(GaussianLikelihood):
        def __init__(self, inv_covar):
            self.inv_covar = inv_covar

    residual = np.array([[1.0, 2.0], [3.0, 4.0]])  # shape (batch=2, n=2)

    def test_exponential_term_full(self):
        inv_covar = np.array([[2.0, 1.0], [1.0, 1.0]])
        dummy = self.Dummy(inv_covar)
        expected = -0.5 * np.array([10.0, 58.0])
        result = dummy._exponential_term_full(self.residual)
        np.testing.assert_allclose(result, expected)

    def test_exponential_term_diagonal(self):
        inv_covar = np.array([2.0, 1.0])
        dummy = self.Dummy(inv_covar)
        expected = -0.5 * np.array([6.0, 34.0])
        result = dummy._exponential_term_diagonal(self.residual)
        np.testing.assert_allclose(result, expected)

    def test_exponential_term_scalar(self):
        inv_covar = np.array([2.0])
        dummy = self.Dummy(inv_covar)
        expected = -0.5 * np.array([10.0, 50.0])
        result = dummy._exponential_term_scalar(self.residual)
        np.testing.assert_allclose(result, expected)


class TestChooseExponentialTermFunction:
    class Dummy(GaussianLikelihood):
        def __init__(self, inv_covar):
            self.inv_covar = inv_covar

        def _exponential_term_full(self, residual: np.ndarray):
            return "full"

        def _exponential_term_diagonal(self, residual: np.ndarray):
            return "diag"

        def _exponential_term_scalar(self, residual: np.ndarray):
            return "scalar"

    def test_choose_exponential_term_function_full(self):
        inv_covar = np.eye(2)
        dummy = self.Dummy(inv_covar)
        fn = dummy._choose_exponential_term_function()
        assert fn(np.zeros(2)) == "full"

    def test_choose_exponential_term_function_diagonal(self):
        inv_covar = np.array([1.0, 2.0])
        dummy = self.Dummy(inv_covar)
        fn = dummy._choose_exponential_term_function()
        assert fn(np.zeros(2)) == "diag"

    def test_choose_exponential_term_function_scalar(self):
        inv_covar = np.array([1.0])
        dummy = self.Dummy(inv_covar)
        fn = dummy._choose_exponential_term_function()
        assert fn(np.zeros(1)) == "scalar"


class TestVectorisedVsScalar:
    """Test that vectorised and scalar evaluations produce identical results."""

    observed_data = np.array([1.0, 2.0, 3.0])
    model_params_single = np.array([0.5, 1.0, 1.5])
    model_params_batch = np.array(
        [
            [0.5, 1.0, 1.5],
            [0.4, 0.9, 1.4],
            [0.6, 1.1, 1.6],
        ]
    )

    def _vectorised_forward_fn(self, model_params: np.ndarray) -> np.ndarray:
        """Forward function that handles both scalar and batch inputs."""
        return model_params * 2.0

    @pytest.mark.parametrize(
        "inv_covar",
        [
            np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]),
            np.array([2.0, 1.5, 1.0]),
            np.array([1.5]),
        ],
        ids=["full_covariance", "diagonal_covariance", "scalar_covariance"],
    )
    def test_scalar_vs_vectorised_single(self, inv_covar):
        """Test scalar and vectorised evaluation with single model parameters."""
        likelihood_scalar = GaussianLikelihood(
            self._vectorised_forward_fn,
            self.observed_data,
            inv_covar,
        )
        likelihood_vectorised = GaussianLikelihood(
            self._vectorised_forward_fn,
            self.observed_data,
            inv_covar,
        )

        result_scalar = likelihood_scalar(self.model_params_single)
        result_vectorised_single = likelihood_vectorised(self.model_params_single)
        np.testing.assert_allclose(result_scalar, result_vectorised_single)

    @pytest.mark.parametrize(
        "inv_covar",
        [
            np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]]),
            np.array([2.0, 1.5, 1.0]),
            np.array([1.5]),
        ],
        ids=["full_covariance", "diagonal_covariance", "scalar_covariance"],
    )
    def test_scalar_vs_vectorised_batch(self, inv_covar):
        """Test scalar and vectorised evaluation with batch of model parameters."""
        likelihood_scalar = GaussianLikelihood(
            self._vectorised_forward_fn,
            self.observed_data,
            inv_covar,
        )
        likelihood_vectorised = GaussianLikelihood(
            self._vectorised_forward_fn,
            self.observed_data,
            inv_covar,
        )

        # Test batch evaluation
        result_vectorised_batch = likelihood_vectorised(self.model_params_batch)
        assert result_vectorised_batch.shape == (3,)

        # Test that batch includes the single result
        scalar_result = np.array(
            [likelihood_scalar(params) for params in self.model_params_batch]
        )
        np.testing.assert_allclose(scalar_result, result_vectorised_batch)

    def test_scalar_return_type(self):
        """Test that scalar mode returns float, not array."""
        inv_covar = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])

        likelihood = GaussianLikelihood(
            self._vectorised_forward_fn,
            self.observed_data,
            inv_covar,
        )

        result = likelihood(self.model_params_single)
        assert isinstance(result, float)

    def test_vectorised_return_type(self):
        """Test that vectorised mode returns array for batch input."""
        inv_covar = np.array([[2.0, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.0]])

        likelihood = GaussianLikelihood(
            self._vectorised_forward_fn,
            self.observed_data,
            inv_covar,
        )

        result = likelihood(self.model_params_batch)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
