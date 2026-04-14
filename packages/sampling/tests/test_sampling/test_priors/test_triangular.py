"""Tests for TriangularPrior."""

import warnings

import numpy as np
import pytest
from sampling.priors import TriangularPrior
from sampling.priors.compound import CompoundPrior
from sampling.priors.triangular import TriangularPriorComponentConfig


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


def test_triangular_logpdf_at_left_boundary_is_neginf(
    valid_triangular_prior: TriangularPrior,
) -> None:
    # x == 2a: pdf == 0, so logpdf must be -inf (no RuntimeWarning)
    params = np.array([0.0, 1.0])  # first dim at left boundary 2a=0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        logp = valid_triangular_prior(params)
    assert logp == -np.inf


def test_triangular_logpdf_at_right_boundary_is_neginf(
    valid_triangular_prior: TriangularPrior,
) -> None:
    # x == 2b: pdf == 0, so logpdf must be -inf (no RuntimeWarning)
    params = np.array([1.0, 2.0])  # second dim at right boundary 2b=2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
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


def test_triangular_gradient_at_left_boundary_is_zero(
    valid_triangular_prior: TriangularPrior,
) -> None:
    # x == 2a: outside support, gradient must be 0 (no inf)
    params = np.array([0.0, 1.0])  # first dim at 2a=0
    grad = valid_triangular_prior.gradient(params)
    assert np.isfinite(grad).all()
    assert grad[0] == 0.0


def test_triangular_gradient_at_right_boundary_is_zero(
    valid_triangular_prior: TriangularPrior,
) -> None:
    # x == 2b: outside support, gradient must be 0 (no inf)
    params = np.array([1.0, 2.0])  # second dim at 2b=2
    grad = valid_triangular_prior.gradient(params)
    assert np.isfinite(grad).all()
    assert grad[1] == 0.0


def test_triangular_gradient_at_midpoint_is_finite(
    valid_triangular_prior: TriangularPrior,
) -> None:
    # x == a+b: kink (mode of the distribution); gradient from the right-segment formula = -1/(b-a)
    params = np.array([1.0, 1.0])  # both dims at mode a+b = 0+1 = 1
    grad = valid_triangular_prior.gradient(params)
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad, -1.0)


def test_triangular_gradient_outside_support_is_zero(
    valid_triangular_prior: TriangularPrior,
) -> None:
    params = np.array([-1.0, 3.0])  # both outside support
    grad = valid_triangular_prior.gradient(params)
    assert np.isfinite(grad).all()
    np.testing.assert_array_equal(grad, [0.0, 0.0])


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


class TestTriangularPriorComponentConfig:
    """Tests for TriangularPriorComponentConfig."""

    def test_init_with_lists(self) -> None:
        config = TriangularPriorComponentConfig(
            lower_bounds=[0.0, 1.0],
            upper_bounds=[2.0, 3.0],
            indices=[0, 1],
        )
        assert config.lower_bounds == [0.0, 1.0]
        assert config.upper_bounds == [2.0, 3.0]
        assert config.indices == [0, 1]
        assert config.type == "triangular"

    def test_init_with_arrays(self) -> None:
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 3.0])
        config = TriangularPriorComponentConfig(
            lower_bounds=lower, upper_bounds=upper, indices=[0, 1]
        )
        np.testing.assert_array_equal(config.lower_bounds, lower)
        np.testing.assert_array_equal(config.upper_bounds, upper)

    def test_to_prior_component(self) -> None:
        from sampling.priors.component import PriorComponent

        config = TriangularPriorComponentConfig(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
            indices=[0, 1],
        )
        component = config.to_prior_component()
        assert isinstance(component, PriorComponent)
        assert isinstance(component.prior_fn, TriangularPrior)
        assert component.n == 2
        np.testing.assert_array_equal(component.indices, [0, 1])

    def test_compound_prior_from_dict_triangular(self) -> None:
        config_dict = {
            "components": [
                {
                    "type": "triangular",
                    "lower_bounds": [0.0, 0.0],
                    "upper_bounds": [1.0, 1.0],
                    "indices": [0, 1],
                }
            ]
        }
        prior = CompoundPrior.from_dict(config_dict)
        assert isinstance(prior, CompoundPrior)
        assert prior.n == 2

    def test_compound_prior_from_dict_evaluates_correctly(self) -> None:
        config_dict = {
            "components": [
                {
                    "type": "triangular",
                    "lower_bounds": [0.0],
                    "upper_bounds": [1.0],
                    "indices": [0],
                }
            ]
        }
        prior = CompoundPrior.from_dict(config_dict)
        # mode of the triangular distribution is at a+b = 0+1 = 1.0, so logpdf=0
        logp = prior(np.array([1.0]))
        assert logp == pytest.approx(0.0)
