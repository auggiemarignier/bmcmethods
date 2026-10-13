"""Tests for linear_gaussian.data_space."""

import numpy as np
import pytest
from linear_gaussian import GaussianComponent
from linear_gaussian.data_space import (
    posterior_cov,
    posterior_mean,
    posterior_predictive_cov,
    posterior_predictive_mean,
)


class TestDataSpacePosterior:
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

        prior_mean = np.array([0.5, -1.0])
        prior_cov = np.diag([2.0, 3.0])
        noise_mean = np.array([0.2, -0.3])
        noise_cov = np.diag([0.5, 1.0])
        expected_cov = prior_cov - prior_cov @ np.linalg.solve(
            prior_cov + noise_cov, prior_cov
        )
        expected_mean = prior_mean + prior_cov @ np.linalg.solve(
            prior_cov + noise_cov, d - prior_mean - noise_mean
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

    def test_singular_data_covariance_raises(self):
        inferred = [
            GaussianComponent(
                A=np.array([[1.0], [0.0]]), mu=np.zeros(1), C=np.ones((1, 1))
            )
        ]
        nuisance = [
            GaussianComponent(A=np.zeros((2, 1)), mu=np.zeros(1), C=np.ones((1, 1)))
        ]

        with pytest.raises(ValueError, match="Matrix is singular"):
            posterior_mean(np.zeros(2), inferred, nuisance)

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
