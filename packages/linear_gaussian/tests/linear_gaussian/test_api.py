"""Tests for linear_gaussian.api."""

import inspect

import numpy as np
import pytest
from linear_gaussian import (
    GaussianComponent,
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
    calc_posterior_predictive_cov,
    calc_posterior_predictive_mean,
    data_space,
    precision,
)
from linear_gaussian.api import _select_strategy


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

    def test_non_vector_d_raises(self):
        d = np.zeros((2, 1))
        gc = self._make_gc()
        with pytest.raises(ValueError, match="1D array"):
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

    def test_predictive_mean_empty_inferred_raises(self):
        d = np.zeros(2)
        nuisance = self._make_gc()
        with pytest.raises(ValueError, match="inferred"):
            calc_posterior_predictive_mean(d, [], [nuisance])

    def test_predictive_cov_empty_inferred_raises(self):
        nuisance = self._make_gc()
        with pytest.raises(ValueError, match="inferred"):
            calc_posterior_predictive_cov([], [nuisance])

    def test_predictive_mean_empty_nuisance_raises(self):
        d = np.zeros(2)
        inferred = self._make_gc()
        with pytest.raises(ValueError, match="nuisance"):
            calc_posterior_predictive_mean(d, [inferred], [])

    def test_predictive_cov_empty_nuisance_raises(self):
        inferred = self._make_gc()
        with pytest.raises(ValueError, match="nuisance"):
            calc_posterior_predictive_cov([inferred], [])


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


class TestStrategyDispatch:
    def test_selects_matching_modules(self):
        assert _select_strategy("precision") is precision
        assert _select_strategy("data") is data_space

    def test_rejects_unknown_strategy(self):
        with pytest.raises(ValueError, match="formulation must be either"):
            _select_strategy("unknown")

    def test_strategies_share_the_same_public_contract(self):
        for name in ("posterior_mean", "posterior_cov"):
            precision_parameters = tuple(
                inspect.signature(getattr(precision, name)).parameters
            )
            data_parameters = tuple(
                inspect.signature(getattr(data_space, name)).parameters
            )
            assert data_parameters == precision_parameters

    def test_posterior_calculations_agree_across_strategies(self):
        inferred = [
            GaussianComponent(
                A=np.array([[1.0, 0.2], [-0.3, 1.1], [0.5, -0.4]]),
                mu=np.array([0.4, -0.2]),
                C=np.array([[1.5, 0.2], [0.2, 0.8]]),
            )
        ]
        nuisance = [
            GaussianComponent(
                A=np.eye(3),
                mu=np.array([0.1, -0.5, 0.2]),
                C=np.diag([0.7, 1.2, 0.9]),
            )
        ]
        d = np.array([0.8, -0.3, 0.5])
        pairs = (
            (
                calc_posterior_mean(d, inferred, nuisance, formulation="data"),
                calc_posterior_mean(d, inferred, nuisance),
            ),
            (
                calc_posterior_cov(inferred, nuisance, formulation="data"),
                calc_posterior_cov(inferred, nuisance),
            ),
        )
        for data_result, precision_result in pairs:
            np.testing.assert_allclose(
                data_result, precision_result, rtol=1e-10, atol=1e-10
            )


class TestPublicCalculations:
    @staticmethod
    def _components():
        inferred = [
            GaussianComponent(
                A=np.array([[1.0]]), mu=np.array([0.5]), C=np.array([[4.0]])
            )
        ]
        nuisance = [
            GaussianComponent(
                A=np.array([[1.0]]), mu=np.array([0.25]), C=np.array([[1.0]])
            )
        ]
        return inferred, nuisance

    def test_log_evidence_public_api(self):
        inferred, nuisance = self._components()
        d = np.array([1.5])

        result = calc_log_evidence(d, inferred, nuisance)

        expected = -0.5 * np.log(2.0 * np.pi * 5.0) - 0.5 * (1.5 - 0.75) ** 2 / 5.0
        np.testing.assert_allclose(result, expected)

    def test_posterior_predictive_mean_dispatches_to_data_strategy(self):
        inferred, nuisance = self._components()
        d = np.array([1.5])

        result = calc_posterior_predictive_mean(
            d, inferred, nuisance, formulation="data"
        )

        # Posterior latent mean is 0.5 + 4/5 * (1.5 - 0.5 - 0.25) = 1.1.
        np.testing.assert_allclose(result, np.array([1.1 + 0.25]))

    def test_posterior_predictive_cov_dispatches_to_data_strategy(self):
        inferred, nuisance = self._components()

        result = calc_posterior_predictive_cov(inferred, nuisance, formulation="data")

        # Latent posterior variance is 4 - 4^2 / (4 + 1) = 0.8.
        np.testing.assert_allclose(result, np.array([[0.8 + 1.0]]))
