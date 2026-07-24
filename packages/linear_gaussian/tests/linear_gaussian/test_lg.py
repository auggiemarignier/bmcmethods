"""Tests for linear_gaussian/lg.py."""

import numpy as np
import pytest
from linear_gaussian import GaussianComponent, calc_log_evidence, calc_posterior_cov, calc_posterior_mean
from linear_gaussian.lg import (
    _block_diag,
    _build_A_I,
    _build_C_I,
    _build_mu_I,
    _calc_C_eta,
    _calc_h,
    _calc_Lambda,
    _calc_mu_eta,
)


# ---------------------------------------------------------------------------
# GaussianComponent validation
# ---------------------------------------------------------------------------


class TestGaussianComponent:
    def test_valid_construction(self):
        gc = GaussianComponent(
            A=np.ones((3, 2)), mu=np.zeros(2), C=np.eye(2)
        )
        assert gc.A.shape == (3, 2)
        assert gc.mu.shape == (2,)
        assert gc.C.shape == (2, 2)

    def test_A_must_be_2d(self):
        with pytest.raises(ValueError, match="A must be a 2D array"):
            GaussianComponent(A=np.ones(3), mu=np.zeros(3), C=np.eye(3))

    def test_mu_must_be_1d(self):
        with pytest.raises(ValueError, match="mu must be a 1D array"):
            GaussianComponent(A=np.eye(3), mu=np.ones((3, 1)), C=np.eye(3))

    def test_C_must_be_2d_square(self):
        with pytest.raises(ValueError, match="C must be a 2D square matrix"):
            GaussianComponent(A=np.eye(3), mu=np.zeros(3), C=np.ones((3, 4)))

    def test_mu_C_dimensions_must_match(self):
        with pytest.raises(ValueError, match="mu and C dimensions are incompatible"):
            GaussianComponent(A=np.ones((3, 2)), mu=np.zeros(3), C=np.eye(2))

    def test_A_columns_must_match_mu(self):
        with pytest.raises(ValueError, match="A columns and mu are incompatible"):
            GaussianComponent(A=np.ones((3, 4)), mu=np.zeros(2), C=np.eye(2))


# ---------------------------------------------------------------------------
# _block_diag
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _calc_mu_eta and _calc_C_eta
# ---------------------------------------------------------------------------


class TestCalcMuEta:
    def test_single_nuisance_identity(self):
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.array([1.0, 2.0]), C=np.eye(2))]
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


# ---------------------------------------------------------------------------
# _build_A_I, _build_mu_I, _build_C_I
# ---------------------------------------------------------------------------


class TestBuildAI:
    def test_single_component(self):
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        inferred = [GaussianComponent(A=A, mu=np.zeros(2), C=np.eye(2))]
        result = _build_A_I(inferred)
        np.testing.assert_array_equal(result, A)

    def test_two_components_concatenated(self):
        A1 = np.array([[1.0], [2.0]])
        A2 = np.array([[3.0], [4.0]])
        i1 = GaussianComponent(A=A1, mu=np.zeros(1), C=np.eye(1))
        i2 = GaussianComponent(A=A2, mu=np.zeros(1), C=np.eye(1))
        result = _build_A_I([i1, i2])
        expected = np.array([[1.0, 3.0], [2.0, 4.0]])
        np.testing.assert_array_equal(result, expected)


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


# ---------------------------------------------------------------------------
# _calc_Lambda and _calc_h
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Input validation for public functions
# ---------------------------------------------------------------------------


class TestInputValidation:
    def _make_gc(self, M=2, N=2):
        return GaussianComponent(A=np.eye(M, N), mu=np.zeros(N), C=np.eye(N))

    def test_both_empty_raises(self):
        d = np.zeros(2)
        with pytest.raises(ValueError, match="At least one"):
            calc_log_evidence(d, [], [])

    def test_inconsistent_M_raises(self):
        d = np.zeros(2)
        i1 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))
        n1 = GaussianComponent(A=np.eye(3), mu=np.zeros(3), C=np.eye(3))
        with pytest.raises(ValueError, match="M"):
            calc_log_evidence(d, [i1], [n1])

    def test_d_dimension_mismatch_raises(self):
        d = np.zeros(3)
        gc = self._make_gc(M=2, N=2)
        with pytest.raises(ValueError, match="d"):
            calc_log_evidence(d, [gc], [])

    def test_posterior_empty_inferred_raises(self):
        d = np.zeros(2)
        n = self._make_gc()
        with pytest.raises(ValueError, match="inferred"):
            calc_posterior_mean(d, [], [n])

    def test_posterior_cov_empty_inferred_raises(self):
        n = self._make_gc()
        with pytest.raises(ValueError, match="inferred"):
            calc_posterior_cov([], [n])

    def test_posterior_empty_nuisance_raises(self):
        d = np.zeros(2)
        i = self._make_gc()
        with pytest.raises(ValueError, match="nuisance"):
            calc_posterior_mean(d, [i], [])

    def test_posterior_cov_empty_nuisance_raises(self):
        i = self._make_gc()
        with pytest.raises(ValueError, match="nuisance"):
            calc_posterior_cov([i], [])


# ---------------------------------------------------------------------------
# calc_posterior_cov
# ---------------------------------------------------------------------------


class TestCalcPosteriorCov:
    def test_1d_known_result(self):
        """1D: sigma_prior=2, sigma_noise=1 => sigma_post = 0.8."""
        inferred = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))]
        nuisance = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))]
        C_post = calc_posterior_cov(inferred, nuisance)
        np.testing.assert_allclose(C_post, np.array([[0.8]]))

    def test_posterior_narrower_than_prior(self):
        """Posterior covariance should be smaller than prior."""
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        C_post = calc_posterior_cov(inferred, nuisance)
        C_prior = np.eye(2)
        # eigenvalues of (C_prior - C_post) should all be non-negative
        eigenvalues = np.linalg.eigvalsh(C_prior - C_post)
        assert np.all(eigenvalues >= -1e-10)

    def test_A_zero_posterior_equals_prior(self):
        """With A_I=0, data is uninformative: posterior covariance = prior."""
        C_prior = np.array([[4.0, 1.0], [1.0, 3.0]])
        inferred = [GaussianComponent(A=np.zeros((2, 2)), mu=np.zeros(2), C=C_prior)]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        C_post = calc_posterior_cov(inferred, nuisance)
        np.testing.assert_allclose(C_post, C_prior, atol=1e-12)

    def test_identity_case(self):
        """A=I, C_I=I, C_eta=I => C_post = 0.5*I."""
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        C_post = calc_posterior_cov(inferred, nuisance)
        np.testing.assert_allclose(C_post, 0.5 * np.eye(2), atol=1e-12)

    def test_two_inferred_components(self):
        """Two inferred components pointing in orthogonal directions.

        A1 = [[1],[0]], A2 = [[0],[1]], C1=C2=4*I1, C_eta=I2.
        => A_I = I_2, C_I = 4*I_2, Lambda = (5/4)*I_2, C_post = (4/5)*I_2.
        """
        i1 = GaussianComponent(
            A=np.array([[1.0], [0.0]]), mu=np.zeros(1), C=np.array([[4.0]])
        )
        i2 = GaussianComponent(
            A=np.array([[0.0], [1.0]]), mu=np.zeros(1), C=np.array([[4.0]])
        )
        n = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))
        C_post = calc_posterior_cov([i1, i2], [n])
        np.testing.assert_allclose(C_post, (4.0 / 5.0) * np.eye(2), atol=1e-12)


# ---------------------------------------------------------------------------
# calc_posterior_mean
# ---------------------------------------------------------------------------


class TestCalcPosteriorMean:
    def test_1d_known_result(self):
        """1D: prior N(0,4), noise N(0,1), d=1.5 => mu_post=1.2."""
        d = np.array([1.5])
        inferred = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))]
        nuisance = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        np.testing.assert_allclose(mu_post, np.array([1.2]))

    def test_zero_data_zero_prior_mean(self):
        """With d=0 and mu_prior=0: posterior mean is 0."""
        d = np.zeros(2)
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        np.testing.assert_allclose(mu_post, np.zeros(2), atol=1e-12)

    def test_A_zero_posterior_mean_equals_prior(self):
        """With A_I=0, data is uninformative: posterior mean = prior mean."""
        mu_prior = np.array([3.0, -1.0])
        d = np.array([100.0, -100.0])
        inferred = [GaussianComponent(A=np.zeros((2, 2)), mu=mu_prior, C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        np.testing.assert_allclose(mu_post, mu_prior, atol=1e-12)

    def test_posterior_mean_between_prior_and_data(self):
        """1D: posterior mean should be a weighted average of prior and data."""
        d = np.array([10.0])
        mu_prior = np.array([0.0])
        inferred = [GaussianComponent(A=np.array([[1.0]]), mu=mu_prior, C=np.array([[1.0]]))]
        nuisance = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        assert mu_prior[0] < mu_post[0] < d[0]

    def test_nuisance_mean_shifts_effective_data(self):
        """Nonzero nuisance mean shifts the effective observation."""
        d = np.array([3.0])
        mu_eta = np.array([2.0])  # nuisance contribution
        inferred = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))]
        nuisance = [GaussianComponent(A=np.array([[1.0]]), mu=mu_eta, C=np.array([[1.0]]))]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        # Effective d = d - mu_eta = 1.0, with C_I=1, C_eta=1: mu_post = 0.5
        np.testing.assert_allclose(mu_post, np.array([0.5]), atol=1e-12)


# ---------------------------------------------------------------------------
# calc_log_evidence
# ---------------------------------------------------------------------------


class TestCalcLogEvidence:
    def test_1d_known_result(self):
        """1D: d ~ N(0, C_I + C_eta) = N(0, 5). log p(1.5) = log N(1.5; 0, 5)."""
        d = np.array([1.5])
        inferred = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))]
        nuisance = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))]
        log_Z = calc_log_evidence(d, inferred, nuisance)
        C_marginal = np.array([[5.0]])
        expected = -0.5 * np.log(2 * np.pi * 5.0) - 0.5 * 1.5**2 / 5.0
        np.testing.assert_allclose(log_Z, expected, rtol=1e-6)

    def test_identity_2d_zero_data(self):
        """2D: d=0, A=I, C_I=I, C_eta=I => d ~ N(0, 2I), log p(0) = -log(4pi)."""
        d = np.zeros(2)
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        log_Z = calc_log_evidence(d, inferred, nuisance)
        expected = -np.log(4 * np.pi)
        np.testing.assert_allclose(log_Z, expected, rtol=1e-6)

    def test_nonzero_prior_mean(self):
        """With nonzero prior mean, marginal mean shifts accordingly."""
        d = np.array([2.0])
        mu_prior = np.array([1.0])
        inferred = [GaussianComponent(A=np.array([[1.0]]), mu=mu_prior, C=np.array([[1.0]]))]
        nuisance = [GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))]
        log_Z = calc_log_evidence(d, inferred, nuisance)
        # d ~ N(A_I mu_I, A_I C_I A_I^T + C_eta) = N(1, 2)
        expected = -0.5 * np.log(4 * np.pi) - (2.0 - 1.0) ** 2 / 4.0
        np.testing.assert_allclose(log_Z, expected, rtol=1e-6)

    def test_two_inferred_components(self):
        """Two inferred components with A_I = [I | I] (stacking)."""
        d = np.zeros(2)
        i1 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))
        i2 = GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        log_Z = calc_log_evidence(d, [i1, i2], nuisance)
        # A_I = hstack([I, I]) = [[1,0,1,0],[0,1,0,1]] (2x4)
        # A_I C_I A_I^T = A_I * I_4 * A_I^T = [[1,0,1,0],[0,1,0,1]] [[1,0,1,0],[0,1,0,1]]^T
        #               = [[2,0],[0,2]] = 2*I_2
        # Total C_marginal = 2*I + I = 3*I
        expected = -np.log(2 * np.pi) - np.log(3)
        np.testing.assert_allclose(log_Z, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# Special cases
# ---------------------------------------------------------------------------


class TestEmptyInferred:
    def test_evidence_equals_nuisance_marginal(self):
        """With inferred=[], evidence is just the nuisance marginal N(d; mu_eta, C_eta)."""
        d = np.zeros(2)
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        log_Z = calc_log_evidence(d, [], nuisance)
        # d ~ N(0, I_2), log p(0) = -log(2*pi)
        expected = -np.log(2 * np.pi)
        np.testing.assert_allclose(log_Z, expected, rtol=1e-6)

    def test_evidence_matches_full_zero_A_case(self):
        """With inferred having A=0, result matches inferred=[] case."""
        d = np.zeros(2)
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        log_Z_empty = calc_log_evidence(d, [], nuisance)

        inferred_zero_A = [GaussianComponent(A=np.zeros((2, 2)), mu=np.zeros(2), C=np.eye(2))]
        log_Z_zero_A = calc_log_evidence(d, inferred_zero_A, nuisance)
        np.testing.assert_allclose(log_Z_empty, log_Z_zero_A, rtol=1e-6)


class TestEmptyNuisance:
    def test_evidence_uses_prior_marginal(self):
        """With nuisance=[], d ~ N(A_I mu_I, A_I C_I A_I^T)."""
        d = np.zeros(2)
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        log_Z = calc_log_evidence(d, inferred, [])
        # d ~ N(0, I_2)
        expected = -np.log(2 * np.pi)
        np.testing.assert_allclose(log_Z, expected, rtol=1e-6)

    def test_empty_inferred_and_empty_nuisance_differ_from_either_alone(self):
        """Empty nuisance gives different evidence than nuisance=[noise_I]."""
        d = np.zeros(2)
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        log_Z_with_nuisance = calc_log_evidence(d, inferred, nuisance)
        log_Z_no_nuisance = calc_log_evidence(d, inferred, [])
        assert not np.isclose(log_Z_with_nuisance, log_Z_no_nuisance)
