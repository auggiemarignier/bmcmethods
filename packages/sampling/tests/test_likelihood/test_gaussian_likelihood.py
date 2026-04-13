"""Test the likelihood functions."""

import pickle
from typing import Self

import numpy as np
import pytest
from sampling.likelihood._base import ForwardBase, ForwardGradientBase
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


class DummyForward(ForwardBase[int]):
    state = 1

    @classmethod
    def from_state(cls, state: int) -> Self:
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        return _dummy_forward_fn(model_params)


@pytest.fixture
def fwd() -> DummyForward:
    return DummyForward()


def _dummy_forward_fn_gradient(model_params: np.ndarray) -> np.ndarray:
    """A simple forward function gradient for testing purposes."""
    if model_params.ndim == 1:
        return 2 * np.eye(model_params.size)
    else:
        return np.stack(
            [2 * np.eye(model_params.shape[1]) for _ in range(model_params.shape[0])],
            axis=0,
        )


class DummyForwardGradient(ForwardGradientBase[int]):
    state = 2

    @classmethod
    def from_state(cls, state: int) -> Self:
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        return _dummy_forward_fn_gradient(model_params)


@pytest.fixture
def fwd_grad() -> DummyForwardGradient:
    return DummyForwardGradient()


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
        observed_data=observed_data,
        inv_covar=request.param,
    )


def test_gaussian_log_likelihood_perfect_match(state: GaussianLikelihoodState) -> None:
    """Test the Gaussian log likelihood function."""
    model_params = (
        state.observed_data / 2.0
    )  # So that predicted data matches observed data
    log_likelihood = gaussian_log_likelihood(model_params, _dummy_forward_fn, state)

    expected_log_likelihood = 0.0  # Perfect match
    assert np.isclose(log_likelihood, expected_log_likelihood)


def test_gaussian_log_likelihood_non_perfect_match(
    state: GaussianLikelihoodState,
) -> None:
    """Test the Gaussian log likelihood function."""
    model_params = state.observed_data
    log_likelihood = gaussian_log_likelihood(model_params, _dummy_forward_fn, state)

    expected_log_likelihood = -0.5 * np.sum(
        state.observed_data**2
    )  # For this particular forward function
    assert np.isclose(log_likelihood, expected_log_likelihood)


def test_grad_gaussian_loglikelihood_at_maximum(state: GaussianLikelihoodState) -> None:
    """Test the Gaussian loglikelihood gradient function at the maximum."""
    model_params = state.observed_data / 2.0
    gradient = grad_gaussian_loglikelihood(
        model_params, _dummy_forward_fn, _dummy_forward_fn_gradient, state
    )

    expected_gradient = np.array(
        [0.0, 0.0, 0.0]
    )  # Gradient should be zero at the maximum likelihood point
    np.testing.assert_allclose(gradient, expected_gradient)


def test_grad_gaussian_loglikelihood_elsewhere(state: GaussianLikelihoodState) -> None:
    """Test the Gaussian loglikelihood gradient function at an easily calculable place."""
    model_params = state.observed_data
    gradient = grad_gaussian_loglikelihood(
        model_params, _dummy_forward_fn, _dummy_forward_fn_gradient, state
    )
    expected_gradient = -2 * state.observed_data
    np.testing.assert_allclose(gradient, expected_gradient)


class TestGaussianLikelihood:
    def test_from_state(
        self,
        state: GaussianLikelihoodState,
        fwd: DummyForward,
        fwd_grad: DummyForwardGradient,
    ) -> None:
        obj = GaussianLikelihood.from_state(
            state, forward=fwd, forward_gradient=fwd_grad
        )

        assert obj.forward is fwd
        assert obj.forward_gradient is fwd_grad

        np.testing.assert_array_equal(obj.state.observed_data, state.observed_data)
        np.testing.assert_array_equal(obj.state.inv_covar, state.inv_covar)
        assert obj.state.covar_kind == state.covar_kind

    def test_call(
        self,
        state: GaussianLikelihoodState,
        fwd: DummyForward,
        fwd_grad: DummyForwardGradient,
    ) -> None:
        likelihood = GaussianLikelihood.from_state(
            state, forward=fwd, forward_gradient=fwd_grad
        )
        model_params = state.observed_data
        expected_log_likelihood = -0.5 * np.sum(
            state.observed_data**2
        )  # For this particular forward function
        assert np.isclose(likelihood(model_params), expected_log_likelihood)

    def test_gradient(
        self,
        state: GaussianLikelihoodState,
        fwd: DummyForward,
        fwd_grad: DummyForwardGradient,
    ) -> None:
        likelihood = GaussianLikelihood.from_state(
            state, forward=fwd, forward_gradient=fwd_grad
        )
        model_params = state.observed_data
        expected_gradient = -2 * state.observed_data
        np.testing.assert_allclose(likelihood.gradient(model_params), expected_gradient)

    def test_gradient_called_without_forward_fn_gradient(
        self, state: GaussianLikelihoodState, fwd: DummyForward
    ) -> None:
        likelihood = GaussianLikelihood(
            forward=fwd,
            observed_data=state.observed_data,
            inv_covar=state.inv_covar,
            forward_gradient=None,
        )
        with pytest.raises(
            RuntimeError,
            match="Gradient function for the forward model must be provided",
        ):
            likelihood.gradient(state.observed_data / 2.0)

    def test_invalid_asymmetric_covariance_matrix(self, fwd: DummyForward) -> None:
        """Test that an asymmetrical covariance matrix raises a ValueError."""
        observed_data = np.array([1.0, 2.0])
        covar = np.array([[1.0, 2.0], [0.0, 1.0]])  # Asymmetric

        with pytest.raises(ValueError, match="Covariance matrix must be symmetric."):
            GaussianLikelihood(fwd, observed_data, covar)

    def test_invalid_non_positive_semidefinite_covariance_matrix(
        self, fwd: DummyForward
    ) -> None:
        """Test that a non-positive semidefinite covariance matrix raises a ValueError."""
        observed_data = np.array([1.0, 2.0])
        covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

        with pytest.raises(
            ValueError, match="Inverse covariance matrix must be positive semidefinite."
        ):
            GaussianLikelihood(fwd, observed_data, covar)

    def test_invalid_data_vector_dimension(self, fwd: DummyForward) -> None:
        """Test that a non-one-dimensional data vector raises a ValueError."""
        observed_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array
        covar = np.array([[1.0, 0.0], [0.0, 1.0]])

        with pytest.raises(ValueError, match="Data vector must be one-dimensional."):
            GaussianLikelihood(fwd, observed_data, covar)

    def test_invalid_forward_function_output_dimension(self, fwd: DummyForward) -> None:
        """Test that a forward function returning incorrect output dimension raises a ValueError.

        This only happens when an example_model is provided to the factory.
        """
        observed_data = np.array([1.0, 2.0])
        covar = np.array([[1.0, 0.0], [0.0, 1.0]])

        class BadForward(ForwardBase[int]):
            state = 1

            @classmethod
            def from_state(cls, state: int) -> Self:
                return cls()

            def __call__(self, model_params: np.ndarray) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])  # wrong length

        with pytest.raises(
            ValueError,
            match="shape",
        ):
            GaussianLikelihood(
                BadForward(), observed_data, covar, example_model=np.array([0.0, 0.0])
            )

        try:
            _ = GaussianLikelihood(
                BadForward(), observed_data, covar
            )  # a bad forward function but no example_model, so no check
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

        try:
            _ = GaussianLikelihood(
                fwd,
                observed_data,
                covar,
                example_model=np.array([0.0, 0.0]),
            )  # valid forward function verification
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")

    def test_invalid_covariance_matrix_size(self, fwd: DummyForward) -> None:
        """Test that a covariance matrix with incorrect size raises a ValueError."""
        observed_data = np.array([1.0, 2.0, 3.0])
        covar = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 instead of 3x3

        with pytest.raises(
            ValueError,
            match="shape",
        ):
            GaussianLikelihood(fwd, observed_data, covar)

    def test_picklable(self, fwd: DummyForward):
        """Test that the GaussianLikelihood object is picklable."""
        observed_data = np.array([1.0, 2.0])
        inv_covar = np.eye(2)
        likelihood = GaussianLikelihood(fwd, observed_data, inv_covar)
        pickled = pickle.dumps(likelihood)
        unpickled = pickle.loads(pickled)
        params = np.array([0.0, 0.0])
        assert np.isclose(likelihood(params), unpickled(params))


class TestValidateCovariance:
    def test_gaussian_likelihood_no_covariance_validation(
        self, monkeypatch, fwd: DummyForward
    ) -> None:
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
        _ = GaussianLikelihood(fwd, observed_data, inv_covar, validate_covariance=False)
        assert not called

    def test_scalar_positive(self) -> None:
        # Should not raise
        _validate_covariance_matrix(np.array([1.0]), 1)

    def test_scalar_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="Variance scalar must be positive"):
            _validate_covariance_matrix(np.array([0.0]), 1)
        with pytest.raises(ValueError, match="Variance scalar must be positive"):
            _validate_covariance_matrix(np.array([-1.0]), 1)

    def test_diagonal_positive(self) -> None:
        # Should not raise
        _validate_covariance_matrix(np.array([1.0, 2.0, 3.0]), 3)

    def test_diagonal_nonpositive(self) -> None:
        with pytest.raises(
            ValueError, match="Covariance diagonal elements must be positive"
        ):
            _validate_covariance_matrix(np.array([1.0, 0.0, 2.0]), 3)
        with pytest.raises(
            ValueError, match="Covariance diagonal elements must be positive"
        ):
            _validate_covariance_matrix(np.array([1.0, -2.0, 2.0]), 3)

    def test_full_valid(self) -> None:
        mat = np.eye(2)
        _validate_covariance_matrix(mat, 2)

    def test_full_wrong_shape(self) -> None:
        mat = np.eye(3)
        with pytest.raises(
            ValueError, match="Covariance matrix must be of shape \\(2, 2\\)"
        ):
            _validate_covariance_matrix(mat, 2)

    def test_full_nonsymmetric(self) -> None:
        mat = np.array([[1.0, 2.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="Covariance matrix must be symmetric"):
            _validate_covariance_matrix(mat, 2)

    def test_full_not_pos_semidefinite(self) -> None:
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

    def test_scalar_vs_batched_batch(
        self, state: GaussianLikelihoodState, fwd: DummyForward
    ) -> None:
        """Test scalar and batched evaluation with batch of model parameters."""
        likelihood = GaussianLikelihood.from_state(state, forward=fwd)

        # Test batch evaluation
        result_batch = likelihood(self.model_params_batch)
        assert result_batch.shape == (3,)

        # Test that batch includes the single result
        result_individual = np.array(
            [likelihood(params) for params in self.model_params_batch]
        )
        np.testing.assert_allclose(result_individual, result_batch)

    def test_scalar_return_type(
        self, state: GaussianLikelihoodState, fwd: DummyForward
    ) -> None:
        """Test that scalar mode returns a 0D NumPy array (scalar array), not a Python float."""
        likelihood = GaussianLikelihood.from_state(state, forward=fwd)

        result = likelihood(self.model_params_single)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 0  # scalar array

    def test_batched_return_type(
        self, state: GaussianLikelihoodState, fwd: DummyForward
    ) -> None:
        """Test that batched mode returns array for batch input."""
        likelihood = GaussianLikelihood.from_state(state, forward=fwd)

        result = likelihood(self.model_params_batch)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
