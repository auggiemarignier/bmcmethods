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
    for attr in kwargs:
        assert hasattr(prior._base, attr)
        np.testing.assert_allclose(getattr(prior._base, attr), kwargs[attr])


def test_invalid_prior_type() -> None:
    """Test that an invalid prior type raises a ValueError."""

    with pytest.raises(ValueError, match="Unknown prior type: invalid_type"):
        WrappedPrior(wrap_bounds=[(-180.0, 180.0)], type="invalid_type")


@pytest.fixture(
    params=[
        (PriorType.GAUSSIAN, {"mean": np.array([0.0]), "inv_covar": np.array([[1.0]])}),
        (
            PriorType.UNIFORM,
            {"lower_bounds": np.array([-180.0]), "upper_bounds": np.array([180.0])},
        ),
    ],
    ids=["gaussian", "uniform"],
)
def wrapped_prior(request) -> WrappedPrior:
    prior_type, kwargs = request.param
    return WrappedPrior(wrap_bounds=[(-180.0, 180.0)], type=prior_type, **kwargs)


def test_model_parameter_wrapping(wrapped_prior: WrappedPrior) -> None:
    """Test that model parameters are correctly wrapped around the specified bounds."""

    model_params_and_wrapped = [
        (-181.0, 179.0),  # Just below lower bound should wrap to just below upper bound
        (180.0, -180.0),  # Just above upper bound should wrap to just above lower bound
    ]
    for model_param, expected_wrapped in model_params_and_wrapped:
        model_params = np.array([model_param])
        wrapped_model_params = wrapped_prior._wrap(model_params)
        expected_wrapped_model_params = np.array([expected_wrapped])

        np.testing.assert_allclose(wrapped_model_params, expected_wrapped_model_params)
        assert wrapped_model_params.shape == model_params.shape


def test_call(wrapped_prior: WrappedPrior) -> None:
    """Test that calling the WrappedPrior returns the log-prior from the base prior, modulo wrapping."""

    model_params = np.array([-181.0])  # This should wrap to 179.0
    log_prior = wrapped_prior(model_params)

    wrapped_params = np.array([179.0])
    expected_log_prior = wrapped_prior._base(
        wrapped_params
    )  # Calculate expected log-prior using the base prior with wrapped parameters

    np.testing.assert_allclose(log_prior, expected_log_prior)
