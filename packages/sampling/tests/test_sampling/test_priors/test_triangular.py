"""Tests for TriangularPrior."""

import numpy as np
import pytest
from sampling.priors import TriangularPrior


@pytest.fixture
def lower() -> np.ndarray:
    return np.array([0.0, 0.0])


@pytest.fixture
def upper() -> np.ndarray:
    return np.array([1.0, 1.0])


@pytest.fixture
def valid_triangular_prior(lower: np.ndarray, upper: np.ndarray) -> TriangularPrior:
    return TriangularPrior(lower, upper)


def test_triangular_prior_n(valid_triangular_prior: TriangularPrior) -> None:
    assert valid_triangular_prior.n == 2


def test_triangular_config_params_expose_bounds(
    valid_triangular_prior: TriangularPrior, lower: np.ndarray, upper: np.ndarray
) -> None:
    cfg = valid_triangular_prior.config_params
    assert isinstance(cfg, list)
    assert len(cfg) == 2
    assert cfg[0] is lower or np.allclose(cfg[0], lower)
    assert cfg[1] is upper or np.allclose(cfg[1], upper)


def test_triangular_midpoint_logpdf(valid_triangular_prior: TriangularPrior) -> None:
    # For a=0, b=1, midpoint x=a+b=1 has pdf=1/(b-a)=1 -> logpdf=0 per component
    params = np.array([1.0, 1.0])
    logp = valid_triangular_prior(params)
    assert np.isfinite(logp)
    assert logp == pytest.approx(0.0)


def test_triangular_out_of_support(valid_triangular_prior: TriangularPrior) -> None:
    params = np.array([-0.1, 0.5])
    logp = valid_triangular_prior(params)
    assert logp == -np.inf


def test_triangular_gradient_interior_points(
    valid_triangular_prior: TriangularPrior,
) -> None:
    # For a=0,b=1: at x=0.5 (left segment) gradient = 1/(x-2a) = 2
    # and at x=1.5 (right segment) gradient = -1/(2b-x) = -2
    params = np.array([[0.5, 1.5]])
    grad = valid_triangular_prior.gradient(params)
    assert grad.shape == params.shape
    np.testing.assert_allclose(grad[0, 0], 2.0)
    np.testing.assert_allclose(grad[0, 1], -2.0)


def test_triangular_sample_shape_and_reproducibility(
    valid_triangular_prior: TriangularPrior,
) -> None:
    rng = np.random.default_rng(2026)
    samples = valid_triangular_prior.sample(10, rng)
    assert samples.shape == (10, valid_triangular_prior.n)

    rng1 = np.random.default_rng(1234)
    rng2 = np.random.default_rng(1234)
    s1 = valid_triangular_prior.sample(5, rng1)
    s2 = valid_triangular_prior.sample(5, rng2)
    np.testing.assert_array_equal(s1, s2)


def test_triangular_batched_consistency(lower: np.ndarray, upper: np.ndarray) -> None:
    prior = TriangularPrior(lower, upper)
    models = [np.array([0.5, 1.5]), np.array([1.0, 1.0]), np.array([0.0, 2.0])]
    individual = np.array([prior(m) for m in models])
    batched = prior(np.array(models))
    np.testing.assert_array_equal(individual, batched)
