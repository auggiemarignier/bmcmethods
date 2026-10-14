"""Tests for linear_gaussian.precision."""

import numpy as np
import pytest
from linear_gaussian import GaussianComponent
from linear_gaussian.precision import (
    _calc_h,
    _calc_Lambda,
    posterior_cov,
    posterior_mean,
    posterior_predictive_cov,
    posterior_predictive_mean,
)


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


class TestPrecisionPosterior:
    def test_posterior_for_multiple_components_with_nonzero_means(self):
        inferred = [
            GaussianComponent(
                A=np.array([[1.0], [0.0]]), mu=np.array([0.5]), C=np.array([[2.0]])
            ),
            GaussianComponent(
                A=np.array([[0.0], [1.0]]), mu=np.array([-1.0]), C=np.array([[3.0]])
            ),
        ]
        nuisance = [
            GaussianComponent(
                A=np.eye(2), mu=np.array([0.2, -0.3]), C=np.diag([0.5, 1.0])
            )
        ]
        d = np.array([1.7, 0.4])

        mean = posterior_mean(d, inferred, nuisance)
        covariance = posterior_cov(inferred, nuisance)

        joint_A = np.eye(2)
        prior_cov = np.diag([2.0, 3.0])
        noise_cov = np.diag([0.5, 1.0])
        prior_mean = np.array([0.5, -1.0])
        noise_mean = np.array([0.2, -0.3])
        expected_cov = np.linalg.solve(
            np.linalg.inv(prior_cov) + joint_A.T @ np.linalg.solve(noise_cov, joint_A),
            np.eye(2),
        )
        expected_mean = expected_cov @ (
            np.linalg.solve(prior_cov, prior_mean)
            + joint_A.T @ np.linalg.solve(noise_cov, d - noise_mean)
        )
        np.testing.assert_allclose(covariance, expected_cov)
        np.testing.assert_allclose(mean, expected_mean)

    def test_zero_forward_operator_returns_prior(self):
        prior_mean = np.array([2.0, -1.0])
        prior_cov = np.array([[2.0, 0.3], [0.3, 1.0]])
        inferred = [GaussianComponent(A=np.zeros((2, 2)), mu=prior_mean, C=prior_cov)]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]

        np.testing.assert_allclose(
            posterior_mean(np.array([9.0, -4.0]), inferred, nuisance), prior_mean
        )
        np.testing.assert_allclose(posterior_cov(inferred, nuisance), prior_cov)

    def test_singular_prior_covariance_raises(self):
        inferred = [
            GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.diag([1.0, 0.0]))
        ]
        nuisance = [GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))]

        with pytest.raises(ValueError, match="Matrix is singular"):
            posterior_cov(inferred, nuisance)

    def test_predictive_wrappers_apply_forward_model_and_nuisance(self):
        inferred = [
            GaussianComponent(
                A=np.array([[2.0]]), mu=np.array([0.5]), C=np.array([[4.0]])
            )
        ]
        nuisance = [
            GaussianComponent(
                A=np.array([[1.0]]), mu=np.array([0.3]), C=np.array([[1.0]])
            )
        ]
        d = np.array([2.0])

        latent_mean = posterior_mean(d, inferred, nuisance)
        latent_cov = posterior_cov(inferred, nuisance)
        np.testing.assert_allclose(
            posterior_predictive_mean(d, inferred, nuisance), 2.0 * latent_mean + 0.3
        )
        np.testing.assert_allclose(
            posterior_predictive_cov(inferred, nuisance), 4.0 * latent_cov + 1.0
        )
