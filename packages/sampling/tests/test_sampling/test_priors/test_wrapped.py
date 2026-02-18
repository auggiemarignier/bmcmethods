"""Tests for priors with wrapped bounds."""

from typing import Any

import numpy as np
import pytest
from sampling.priors import GaussianPrior, UniformPrior, WrappedPrior
from sampling.priors._protocols import PriorFunction, PriorType


@pytest.mark.parametrize(
    "prior_type,expected_class,kwargs",
    [
        (
            PriorType.GAUSSIAN,
            GaussianPrior,
            {"mean": np.array([0.0]), "inv_covar": np.array([[1.0]])},
        ),
        (
            PriorType.UNIFORM,
            UniformPrior,
            {"lower_bounds": np.array([-180.0]), "upper_bounds": np.array([180.0])},
        ),
    ],
)
def test_valid_initialisation(
    prior_type: PriorType, expected_class: type[PriorFunction], kwargs: dict[str, Any]
) -> None:
    """Test that the chosen prior type is correct."""

    prior = WrappedPrior(wrap_bounds=[(-180.0, 180.0)], type=prior_type, **kwargs)

    assert prior.wrap_bounds == [(-180.0, 180.0)]
    assert prior.type == prior_type
    assert isinstance(prior._base, expected_class)


def test_invalid_prior_type() -> None:
    """Test that an invalid prior type raises a ValueError."""

    with pytest.raises(ValueError, match="Unknown prior type: invalid_type"):
        WrappedPrior(wrap_bounds=[(-180.0, 180.0)], type="invalid_type")


@pytest.mark.parametrize(
    "prior_type,kwargs",
    [
        (
            PriorType.GAUSSIAN,
            {"mean": np.array([0.0]), "inv_covar": np.array([[1.0]])},
        ),
        (
            PriorType.UNIFORM,
            {"lower_bounds": np.array([-180.0]), "upper_bounds": np.array([180.0])},
        ),
    ],
)
def test_model_parameter_wrapping(
    prior_type: PriorType, kwargs: dict[str, Any]
) -> None:
    """Test that model parameters are correctly wrapped around the specified bounds."""

    prior = WrappedPrior(wrap_bounds=[(-180.0, 180.0)], type=prior_type, **kwargs)

    # Test wrapping for a parameter just outside the lower bound
    model_params = np.array([-181.0])  # This should wrap to 179.0
    wrapped_model_params = prior._wrap(model_params)
    expected_model_params = np.array([179.0])
    np.testing.assert_allclose(wrapped_model_params, expected_model_params)
    assert wrapped_model_params.shape == model_params.shape

    # Test wrapping for a parameter just outside the upper bound
    model_params = np.array([181.0])  # This should wrap to -179.0
    wrapped_model_params = prior._wrap(model_params)
    expected_model_params = np.array([-179.0])
    np.testing.assert_allclose(wrapped_model_params, expected_model_params)
    assert wrapped_model_params.shape == model_params.shape


@pytest.mark.parametrize(
    "prior_type,kwargs",
    [
        (
            PriorType.GAUSSIAN,
            {"mean": np.array([0.0]), "inv_covar": np.array([[1.0]])},
        ),
        (
            PriorType.UNIFORM,
            {"lower_bounds": np.array([-180.0]), "upper_bounds": np.array([180.0])},
        ),
    ],
)
def test_call(prior_type: PriorType, kwargs: dict[str, Any]) -> None:
    """Test that calling the WrappedPrior returns the log-prior from the base prior, modulo wrapping."""

    prior = WrappedPrior(
        wrap_bounds=[(-180.0, 180.0)],
        type=prior_type,
        **kwargs,
    )

    model_params = np.array([-181.0])  # This should wrap to 179.0
    log_prior = prior(model_params)

    wrapped_params = np.array([179.0])
    expected_log_prior = prior._base(
        wrapped_params
    )  # Calculate expected log-prior using the base prior with wrapped parameters

    np.testing.assert_allclose(log_prior, expected_log_prior)
