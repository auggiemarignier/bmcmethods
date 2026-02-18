"""Tests for priors with wrapped bounds."""

import inspect
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


@pytest.mark.parametrize(
    "model_param,expected_wrapped",
    [
        (-181.0, 179.0),  # Just below lower bound should wrap to just below upper bound
        (181.0, -179.0),  # Just above upper bound should wrap to just above lower bound
        (180.0, -180.0),  # Exactly at upper bound should wrap to lower bound
        (-180.0, -180.0),  # Exactly at lower bound should stay the same
        (360.0, 0.0),  # Exactly one full range above should wrap to the same point
        (-360.0, 0.0),  # Exactly one full range below should wrap to the same point
        (540.0, -180.0),  # One and a half ranges above should wrap to lower bound
        (0.0, 0.0),  # Within bounds should not change
    ],
)
def test_model_parameter_wrapping(
    wrapped_prior: WrappedPrior, model_param: float, expected_wrapped: float
) -> None:
    """Test that model parameters are correctly wrapped around the specified bounds."""

    model_params = np.array([model_param])
    wrapped_model_params = wrapped_prior._wrap(model_params)
    expected_wrapped_model_params = np.array([expected_wrapped])

    np.testing.assert_allclose(wrapped_model_params, expected_wrapped_model_params)
    assert wrapped_model_params.shape == model_params.shape


def test_model_parameter_wrapping_batch(wrapped_prior: WrappedPrior) -> None:
    """Test that model parameters are correctly wrapped for a batch of parameters."""

    model_params = np.array([[-181.0], [180.0], [360.0], [-360.0], [540.0], [0.0]])
    expected_wrapped_params = np.array(
        [[179.0], [-180.0], [0.0], [0.0], [-180.0], [0.0]]
    )

    wrapped_model_params = wrapped_prior._wrap(model_params)

    np.testing.assert_allclose(wrapped_model_params, expected_wrapped_params)
    assert wrapped_model_params.shape == model_params.shape


def test_multidimensional_batched_wrapping(wrapped_prior: WrappedPrior) -> None:
    """Test that multidimensional model parameters are correctly wrapped for a batch of parameters."""

    # patch the fixture to have 2D wrap bounds for this test
    wrap_bounds = [(-180.0, 180.0), (-90.0, 90.0)]

    def _filter_base_init_kwargs(base) -> dict:
        """Return a dict of attributes from ``base.__dict__`` that match the
        parameter names of ``base.__class__.__init__`` (excluding ``self``).

        This avoids passing derived attributes stored on the instance that are
        not actual constructor parameters.
        """
        sig = inspect.signature(base.__class__.__init__)
        valid_names = [
            name
            for name, param in sig.parameters.items()
            if name != "self"
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        return {k: v for k, v in base.__dict__.items() if k in valid_names}

    base_kwargs = _filter_base_init_kwargs(wrapped_prior._base)
    wrapped_prior = WrappedPrior(
        wrap_bounds=wrap_bounds, type=wrapped_prior.type, **base_kwargs
    )

    model_params = np.array(
        [[-181.0, -91.0], [180.0, 90.0], [360.0, 180.0], [-360.0, -180.0]]
    )
    expected_wrapped_params = np.array(
        [[179.0, 89.0], [-180.0, -90.0], [0.0, 0.0], [0.0, 0.0]]
    )

    wrapped_model_params = wrapped_prior._wrap(model_params)

    np.testing.assert_allclose(wrapped_model_params, expected_wrapped_params)
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


def test_sample(wrapped_prior: WrappedPrior) -> None:
    """Test that we can sample from the WrappedPrior."""

    rng = np.random.default_rng(seed=42)
    num_samples = 1000
    samples = wrapped_prior.sample(num_samples=num_samples, rng=rng)

    assert samples.shape == (num_samples, len(wrapped_prior.wrap_bounds))
