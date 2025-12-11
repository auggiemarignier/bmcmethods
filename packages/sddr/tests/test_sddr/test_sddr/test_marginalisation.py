"""Tests for prior marginalisation."""

from collections.abc import Callable

import numpy as np
import pytest
from sampling.priors import (
    CompoundPrior,
    GaussianPrior,
    PriorComponent,
    PriorFunction,
    UniformPrior,
)
from sddr.marginalisation import marginalise_prior


@pytest.fixture
def gaussian_prior() -> Callable[[np.ndarray], float]:
    """Create a Gaussian prior for testing."""
    mean = np.array([0.0, 0.0, 0.0])
    covar = np.eye(3)
    return GaussianPrior(mean, covar)


@pytest.fixture
def uniform_prior() -> Callable[[np.ndarray], float]:
    """Create a Uniform prior for testing."""
    lower = np.array([-1.0, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0])
    return UniformPrior(lower, upper)


@pytest.mark.parametrize(
    ("fixture_name", "return_type"),
    [("gaussian_prior", GaussianPrior), ("uniform_prior", UniformPrior)],
)
def test_marginalise_return_type(
    fixture_name: str,
    return_type: type[PriorFunction],
    request: pytest.FixtureRequest,
) -> None:
    """Test that marginalisation returns a callable prior function."""
    prior_fn = request.getfixturevalue(fixture_name)
    assert isinstance(marginalise_prior(prior_fn, [0, 2]), return_type)


@pytest.mark.parametrize(
    ("fixture_name", "expected_result"),
    [("gaussian_prior", -1.0), ("uniform_prior", 0.0)],
)
def test_marginalise_log_prior_computation(
    fixture_name: str, expected_result: float, request: pytest.FixtureRequest
) -> None:
    """Test that marginalised prior computes correct log-prior values."""
    prior_fn = request.getfixturevalue(fixture_name)
    marginalised_prior = marginalise_prior(prior_fn, [0, 2])
    test_point = np.array([1.0, 1.0])

    log_prior = marginalised_prior(test_point)
    np.testing.assert_almost_equal(log_prior, expected_result)


@pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
def test_marginalise_invalid_indices(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test that invalid indices raise ValueError."""
    prior_fn = request.getfixturevalue(fixture_name)

    with pytest.raises(IndexError, match="out of bounds"):
        marginalise_prior(prior_fn, [0, 5])  # 5 is out of bounds


@pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
def test_marginalise_slice(fixture_name: str, request: pytest.FixtureRequest) -> None:
    """Test that a slice gives the same as a list of indices."""
    prior_fn = request.getfixturevalue(fixture_name)
    marginalised_prior_list = marginalise_prior(prior_fn, [0, 2])
    marginalised_prior_slice = marginalise_prior(prior_fn, slice(0, 3, 2))

    test_point = np.array([1.0, 1.0])
    log_prior_list = marginalised_prior_list(test_point)
    log_prior_slice = marginalised_prior_slice(test_point)
    np.testing.assert_almost_equal(log_prior_list, log_prior_slice)


@pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
def test_marginalise_all_indices(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test that marginalising all indices returns the original prior."""
    prior_fn = request.getfixturevalue(fixture_name)
    n_dims = len(
        prior_fn.config_params[0]
    )  # Assuming first config param is mean/lower bound
    all_indices = list(range(n_dims))

    marginalised_prior = marginalise_prior(prior_fn, all_indices)

    test_point = np.array([0.5] * n_dims)
    log_prior_original = prior_fn(test_point)
    log_prior_marginalised = marginalised_prior(test_point)

    np.testing.assert_almost_equal(log_prior_original, log_prior_marginalised)


@pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
def test_marginalise_no_indices(
    fixture_name: str, request: pytest.FixtureRequest
) -> None:
    """Test that marginalising over all variables raises a ValueError."""
    prior_fn = request.getfixturevalue(fixture_name)

    with pytest.raises(ValueError, match="At least one index"):
        marginalise_prior(prior_fn, np.array([]))


def test_compound_prior_marginalisation() -> None:
    """Test marginalisation of the compound prior."""
    # Gaussian prior on first two parameters
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    gaussian_prior = GaussianPrior(mean, covar)
    gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

    # Uniform prior on last two parameters
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    uniform_prior = UniformPrior(lower, upper)
    uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

    # Combine into compound prior
    compound_prior = CompoundPrior([gaussian_component, uniform_component])

    # Marginalise over the first and last parameters (one from Gaussian, one from Uniform)
    marginalised_prior = marginalise_prior(compound_prior, [1, 2])

    # test point 1 stdev away in Gaussian component, and within Uniform component
    test_point = np.array([1.0, 0.0])

    expected_log_prior = -0.5 * (1.0**2) + 0.0  # Gaussian part + Uniform part
    log_prior = marginalised_prior(test_point)
    np.testing.assert_almost_equal(log_prior, expected_log_prior)


def test_compound_prior_marginalisation_empty_indices() -> None:
    """Test that marginalising compound prior with empty indices raises ValueError."""
    # Gaussian prior on first two parameters
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    gaussian_prior = GaussianPrior(mean, covar)
    gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

    # Uniform prior on last two parameters
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    uniform_prior = UniformPrior(lower, upper)
    uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

    # Combine into compound prior
    compound_prior = CompoundPrior([gaussian_component, uniform_component])

    # Try to marginalise with empty indices array
    with pytest.raises(
        ValueError, match="At least one index should be kept after marginalisation"
    ):
        marginalise_prior(compound_prior, np.array([]))


def test_compound_prior_marginalisation_skip_component() -> None:
    """Test marginalising compound prior where one component is completely skipped."""
    # Gaussian prior on first two parameters
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    gaussian_prior = GaussianPrior(mean, covar)
    gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

    # Uniform prior on next two parameters
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    uniform_prior = UniformPrior(lower, upper)
    uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

    # Another Gaussian prior on last parameter
    mean2 = np.array([5.0])
    covar2 = np.array([[2.0]])
    gaussian_prior2 = GaussianPrior(mean2, covar2)
    gaussian_component2 = PriorComponent(prior_fn=gaussian_prior2, indices=[4])

    # Combine into compound prior
    compound_prior = CompoundPrior(
        [gaussian_component, uniform_component, gaussian_component2]
    )

    # Marginalise to keep only indices from first and third components, skipping the uniform component
    marginalised_prior = marginalise_prior(compound_prior, [0, 4])

    # The marginalised prior should have 2 parameters
    assert marginalised_prior.n == 2

    # Test evaluation: first parameter from first Gaussian, second from third Gaussian
    test_point = np.array([1.0, 5.0])  # 1 stdev away from first mean, at second mean
    log_prior = marginalised_prior(test_point)

    expected_log_prior = (
        -0.5 * (1.0**2) + 0.0
    )  # First Gaussian + second Gaussian at mean
    np.testing.assert_almost_equal(log_prior, expected_log_prior)


def test_compound_prior_marginalisation_no_components_remain() -> None:
    """Test marginalising compound prior where no components remain after marginalisation."""
    # Gaussian prior on first two parameters
    mean = np.array([0.0, 0.0])
    covar = np.eye(2)
    gaussian_prior = GaussianPrior(mean, covar)
    gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

    # Uniform prior on next two parameters
    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    uniform_prior = UniformPrior(lower, upper)
    uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

    # Combine into compound prior
    compound_prior = CompoundPrior([gaussian_component, uniform_component])

    # Try to marginalise with indices that don't exist in any component
    with pytest.raises(
        ValueError, match="No prior components remain after marginalisation"
    ):
        marginalise_prior(compound_prior, [5, 6])
