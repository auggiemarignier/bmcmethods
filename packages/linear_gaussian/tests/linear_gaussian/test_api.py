"""Tests for linear_gaussian.api."""

import numpy as np
import pytest
from linear_gaussian import (
    GaussianComponent,
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
    calc_posterior_predictive_cov,
    calc_posterior_predictive_mean,
)


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


class TestCalcPosteriorCov:
    def test_1d_known_result(self):
        """1D: sigma_prior=2, sigma_noise=1 => sigma_post = 0.8."""
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
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

    def test_singular_noise_covariance_raises_clear_error(self):
        """Singular nuisance covariances should fail instead of inverting."""
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [
            GaussianComponent(
                A=np.eye(2),
                mu=np.zeros(2),
                C=np.array([[1.0, 0.0], [0.0, 0.0]]),
            )
        ]

        with pytest.raises(
            ValueError,
            match="Matrix is singular or not numerically positive definite",
        ):
            calc_posterior_cov(inferred, nuisance)


class TestCalcPosteriorMean:
    def test_1d_known_result(self):
        """1D: prior N(0,4), noise N(0,1), d=1.5 => mu_post=1.2."""
        d = np.array([1.5])
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
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
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=mu_prior, C=np.array([[1.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        assert mu_prior[0] < mu_post[0] < d[0]

    def test_nuisance_mean_shifts_effective_data(self):
        """Nonzero nuisance mean shifts the effective observation."""
        d = np.array([3.0])
        mu_eta = np.array([2.0])  # nuisance contribution
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=mu_eta, C=np.array([[1.0]]))
        ]
        mu_post = calc_posterior_mean(d, inferred, nuisance)
        # Effective d = d - mu_eta = 1.0, with C_I=1, C_eta=1: mu_post = 0.5
        np.testing.assert_allclose(mu_post, np.array([0.5]), atol=1e-12)

    def test_ill_conditioned_spd_inputs_remain_finite(self):
        """Ill-conditioned SPD inputs should still yield a finite posterior mean."""
        d = np.array([1e-6, 1.0])
        inferred = [
            GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.diag([1e-12, 1.0]))
        ]
        nuisance = [
            GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.diag([1.0, 1e12]))
        ]

        mu_post = calc_posterior_mean(d, inferred, nuisance)

        expected = np.array([1e-6 / (1.0 + 1e12), 1.0 / (1e12 + 1.0)])
        assert np.all(np.isfinite(mu_post))
        np.testing.assert_allclose(mu_post, expected, rtol=1e-10, atol=1e-18)


class TestCalcPosteriorPredictive:
    def test_1d_known_result(self):
        d = np.array([1.5])
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
        mu_pred = calc_posterior_predictive_mean(d, inferred, nuisance)
        C_pred = calc_posterior_predictive_cov(inferred, nuisance)
        np.testing.assert_allclose(mu_pred, np.array([1.2]), atol=1e-12)
        np.testing.assert_allclose(C_pred, np.array([[1.8]]), atol=1e-12)

    def test_identity_case(self):
        inferred = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]
        d = np.zeros(2)
        mu_pred = calc_posterior_predictive_mean(d, inferred, nuisance)
        C_pred = calc_posterior_predictive_cov(inferred, nuisance)
        np.testing.assert_allclose(mu_pred, np.zeros(2), atol=1e-12)
        np.testing.assert_allclose(C_pred, 1.5 * np.eye(2), atol=1e-12)

    def test_nuisance_mean_shifts_predictive_mean(self):
        d = np.array([3.0])
        mu_eta = np.array([2.0])
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=mu_eta, C=np.array([[1.0]]))
        ]
        mu_pred = calc_posterior_predictive_mean(d, inferred, nuisance)
        # Effective d = d - mu_eta = 1.0 => mu_post = 0.5 => mu_pred = 0.5 + mu_eta = 2.5
        np.testing.assert_allclose(mu_pred, np.array([2.5]), atol=1e-12)

    def test_empty_inferred_raises(self):
        d = np.zeros(1)
        n = GaussianComponent(A=np.eye(1), mu=np.zeros(1), C=np.eye(1))
        with pytest.raises(ValueError, match="inferred"):
            calc_posterior_predictive_mean(d, [], [n])
        with pytest.raises(ValueError, match="inferred"):
            calc_posterior_predictive_cov([], [n])

    def test_empty_nuisance_raises(self):
        d = np.zeros(1)
        i = GaussianComponent(A=np.eye(1), mu=np.zeros(1), C=np.eye(1))
        with pytest.raises(ValueError, match="nuisance"):
            calc_posterior_predictive_mean(d, [i], [])
        with pytest.raises(ValueError, match="nuisance"):
            calc_posterior_predictive_cov([i], [])


class TestCalcLogEvidence:
    def test_1d_known_result(self):
        """1D: d ~ N(0, C_I + C_eta) = N(0, 5). log p(1.5) = log N(1.5; 0, 5)."""
        d = np.array([1.5])
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[4.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
        log_Z = calc_log_evidence(d, inferred, nuisance)
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
        inferred = [
            GaussianComponent(A=np.array([[1.0]]), mu=mu_prior, C=np.array([[1.0]]))
        ]
        nuisance = [
            GaussianComponent(A=np.array([[1.0]]), mu=np.zeros(1), C=np.array([[1.0]]))
        ]
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

        inferred_zero_A = [
            GaussianComponent(A=np.zeros((2, 2)), mu=np.zeros(2), C=np.eye(2))
        ]
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
