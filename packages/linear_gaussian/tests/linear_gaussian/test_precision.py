"""Tests for linear_gaussian.precision."""

import warnings

import numpy as np
import pytest
from linear_gaussian import GaussianComponent
from linear_gaussian.precision import (
    _block_diag,
    _build_A_I,
    _build_C_I,
    _build_mu_I,
    _calc_C_eta,
    _calc_h,
    _calc_Lambda,
    _calc_mu_eta,
    _log_gaussian_density,
    _solve_symmetric_system,
)


class TestBlockDiag:
    def test_single_matrix(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(_block_diag([A]), A)

    def test_two_matrices_shape(self):
        A = np.eye(2)
        B = np.eye(3)
        result = _block_diag([A, B])
        assert result.shape == (5, 5)

    def test_two_matrices_values(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0]])
        result = _block_diag([A, B])
        expected = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        np.testing.assert_array_equal(result, expected)


class TestCalcMuEta:
    def test_single_nuisance_identity(self):
        nuisance = [
            GaussianComponent(A=np.eye(2), mu=np.array([1.0, 2.0]), C=np.eye(2))
        ]
        result = _calc_mu_eta(nuisance, M=2)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_two_nuisance_components(self):
        n1 = GaussianComponent(A=np.eye(2), mu=np.array([1.0, 0.0]), C=np.eye(2))
        n2 = GaussianComponent(A=np.eye(2), mu=np.array([0.0, 2.0]), C=np.eye(2))
        result = _calc_mu_eta([n1, n2], M=2)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0]))

    def test_empty_nuisance_returns_zero_vector(self):
        result = _calc_mu_eta([], M=3)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_nontrivial_A(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        mu = np.array([1.0, 1.0])
        nuisance = [GaussianComponent(A=A, mu=mu, C=np.eye(2))]
        result = _calc_mu_eta(nuisance, M=2)
        np.testing.assert_array_equal(result, np.array([2.0, 3.0]))


class TestCalcCEta:
    def test_single_nuisance_identity(self):
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        result = _calc_C_eta(nuisance, M=2)
        np.testing.assert_array_equal(result, np.eye(2))

    def test_two_nuisance_components_sum(self):
        n1 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))
        n2 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))
        result = _calc_C_eta([n1, n2], M=2)
        np.testing.assert_array_equal(result, 2.0 * np.eye(2))

    def test_empty_nuisance_returns_zero_matrix(self):
        result = _calc_C_eta([], M=2)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))

    def test_nontrivial_A(self):
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        C = np.eye(2)
        nuisance = [GaussianComponent(A=A, mu=np.zeros(2), C=C)]
        result = _calc_C_eta(nuisance, M=2)
        expected = A @ C @ A.T
        np.testing.assert_array_equal(result, expected)


class TestBuildAI:
    def test_single_component(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        inferred = [GaussianComponent(A=A, mu=np.zeros(2), C=np.eye(2))]
        result = _build_A_I(inferred, 2)
        np.testing.assert_array_equal(result, A)

    def test_two_components_concatenated(self):
        A1 = np.array([[1.0], [2.0]])
        A2 = np.array([[3.0], [4.0]])
        i1 = GaussianComponent(A=A1, mu=np.zeros(1), C=np.eye(1))
        i2 = GaussianComponent(A=A2, mu=np.zeros(1), C=np.eye(1))
        result = _build_A_I([i1, i2], 2)
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_empty_inferred_returns_zero_columns(self):
        result = _build_A_I([], 3)
        assert result.shape == (3, 0)


class TestBuildMuI:
    def test_single_component(self):
        mu = np.array([1.0, 2.0])
        inferred = [GaussianComponent(A=np.eye(2), mu=mu, C=np.eye(2))]
        result = _build_mu_I(inferred)
        np.testing.assert_array_equal(result, mu)

    def test_two_components_stacked(self):
        mu1 = np.array([1.0, 2.0])
        mu2 = np.array([3.0])
        i1 = GaussianComponent(A=np.eye(2), mu=mu1, C=np.eye(2))
        i2 = GaussianComponent(A=np.ones((2, 1)), mu=mu2, C=np.eye(1))
        result = _build_mu_I([i1, i2])
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))


class TestBuildCI:
    def test_single_component(self):
        C = np.array([[4.0, 1.0], [1.0, 3.0]])
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=C)]
        result = _build_C_I(inferred)
        np.testing.assert_array_equal(result, C)

    def test_two_components_block_diagonal(self):
        C1 = np.array([[2.0, 0.0], [0.0, 3.0]])
        C2 = np.array([[5.0]])
        i1 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=C1)
        i2 = GaussianComponent(A=np.ones((2, 1)), mu=np.zeros(1), C=C2)
        result = _build_C_I([i1, i2])
        expected = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 5.0]])
        np.testing.assert_array_equal(result, expected)

    def test_off_diagonal_blocks_are_zero(self):
        C1 = np.array([[1.0, 0.5], [0.5, 1.0]])
        C2 = np.array([[2.0, 0.1], [0.1, 2.0]])
        i1 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=C1)
        i2 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=C2)
        result = _build_C_I([i1, i2])
        np.testing.assert_array_equal(result[:2, 2:], np.zeros((2, 2)))
        np.testing.assert_array_equal(result[2:, :2], np.zeros((2, 2)))


class TestCalcLambda:
    def test_1d_known_result(self):
        """1D case: Lambda = 1/C_I + A^T/C_eta*A = 0.25 + 1 = 1.25."""
        C_I = np.array([[4.0]])
        A_I = np.array([[1.0]])
        C_eta = np.array([[1.0]])
        result = _calc_Lambda(C_I, A_I, C_eta)
        np.testing.assert_allclose(result, np.array([[1.25]]))

    def test_identity_case(self):
        """With C_I=I, A_I=I, C_eta=I: Lambda = I + I = 2*I."""
        C_I = np.eye(2)
        A_I = np.eye(2)
        C_eta = np.eye(2)
        result = _calc_Lambda(C_I, A_I, C_eta)
        np.testing.assert_allclose(result, 2.0 * np.eye(2))

    def test_zero_A_reduces_to_prior_precision(self):
        """With A_I=0: Lambda = C_I^{-1}."""
        C_I = np.array([[4.0, 0.0], [0.0, 2.0]])
        A_I = np.zeros((2, 2))
        C_eta = np.eye(2)
        result = _calc_Lambda(C_I, A_I, C_eta)
        np.testing.assert_allclose(result, np.linalg.inv(C_I))

    def test_ill_conditioned_spd_inputs_remain_finite(self):
        """Ill-conditioned SPD covariances should still produce a finite precision."""
        C_I = np.diag([1e-12, 1.0])
        A_I = np.eye(2)
        C_eta = np.diag([1.0, 1e12])

        result = _calc_Lambda(C_I, A_I, C_eta)

        expected = np.array([[1e12 + 1.0, 0.0], [0.0, 1.0 + 1e-12]])
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


class TestCalcH:
    def test_1d_known_result(self):
        """1D case with d=1.5, mu_I=0, C_I=4, A_I=1, mu_eta=0, C_eta=1."""
        d = np.array([1.5])
        mu_I = np.array([0.0])
        C_I = np.array([[4.0]])
        A_I = np.array([[1.0]])
        mu_eta = np.array([0.0])
        C_eta = np.array([[1.0]])
        result = _calc_h(d, mu_I, C_I, A_I, mu_eta, C_eta)
        np.testing.assert_allclose(result, np.array([1.5]))

    def test_zero_data_contribution(self):
        """With A_I=0 and mu_eta=0: h = C_I^{-1} mu_I."""
        d = np.array([5.0, 3.0])
        mu_I = np.array([1.0, 2.0])
        C_I = np.eye(2)
        A_I = np.zeros((2, 2))
        mu_eta = np.zeros(2)
        C_eta = np.eye(2)
        result = _calc_h(d, mu_I, C_I, A_I, mu_eta, C_eta)
        np.testing.assert_allclose(result, mu_I)


class TestSolveSymmetricSystem:
    def test_matches_dense_solve_without_inv(self, monkeypatch):
        """The stabilized solve path should not rely on np.linalg.inv."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        rhs = np.array([1.0, -2.0])

        def fail_inv(_: np.ndarray) -> np.ndarray:
            raise AssertionError("np.linalg.inv should not be used")

        monkeypatch.setattr(np.linalg, "inv", fail_inv)

        result = _solve_symmetric_system(None, matrix, rhs)

        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(result, expected)

    def test_uses_precomputed_cholesky_without_refactorizing(self, monkeypatch):
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        rhs = np.array([1.0, -2.0])
        L = np.linalg.cholesky(matrix)

        def fail_cholesky(_: np.ndarray) -> np.ndarray:
            raise AssertionError("np.linalg.cholesky should not be used")

        monkeypatch.setattr(np.linalg, "cholesky", fail_cholesky)

        result = _solve_symmetric_system(L, matrix, rhs)

        expected = np.linalg.solve(matrix, rhs)
        np.testing.assert_allclose(result, expected)

    def test_singular_matrix_raises_clear_error(self):
        """Singular covariance/precision matrices should raise a clear error."""
        matrix = np.array([[1.0, 0.0], [0.0, 0.0]])
        rhs = np.array([1.0, 2.0])

        with pytest.raises(
            ValueError,
            match="Matrix is singular or not numerically positive definite",
        ):
            _solve_symmetric_system(None, matrix, rhs)


class TestLogGaussianDensity:
    @staticmethod
    def _make_ill_conditioned_cov(
        m: int = 250, n: int = 20, scale: float = 1e20
    ) -> np.ndarray:
        """Construct a large, ill-conditioned SPD covariance matrix."""
        rng = np.random.default_rng(0)
        A = rng.normal(size=(m, n))
        return A @ (scale * np.eye(n)) @ A.T + np.eye(m)

    def test_avoids_slogdet_for_ill_conditioned_spd_covariance(self, monkeypatch):
        """Ill-conditioned SPD matrices should not require slogdet."""
        cov = self._make_ill_conditioned_cov()
        x = np.zeros(cov.shape[0])
        mean = np.zeros_like(x)

        def fail_slogdet(_: np.ndarray) -> tuple[float, float]:
            raise AssertionError("slogdet should not be used for SPD covariances")

        monkeypatch.setattr(np.linalg, "slogdet", fail_slogdet)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            log_density = _log_gaussian_density(x, mean, cov)

        assert np.isfinite(log_density)
        assert not any(isinstance(w.message, RuntimeWarning) for w in caught)

    def test_slogdet_fallback_when_cholesky_fails(self, monkeypatch):
        """When both Cholesky attempts fail, slogdet is used and returns a finite result."""
        cov = np.diag([1.0, 2.0, 3.0])
        x = np.array([0.5, 1.0, -0.5])
        mean = np.zeros(3)

        original_cholesky = np.linalg.cholesky

        def always_fail(a: np.ndarray) -> np.ndarray:
            raise np.linalg.LinAlgError("forced failure")

        monkeypatch.setattr(np.linalg, "cholesky", always_fail)

        log_density = _log_gaussian_density(x, mean, cov)

        # Restore to compute reference value
        monkeypatch.setattr(np.linalg, "cholesky", original_cholesky)
        diff = x - mean
        sign, log_det = np.linalg.slogdet(cov)
        expected = float(
            -0.5 * 3 * np.log(2 * np.pi)
            - 0.5 * log_det
            - 0.5 * float(diff @ np.linalg.solve(cov, diff))
        )

        assert np.isfinite(log_density)
        np.testing.assert_allclose(log_density, expected)
