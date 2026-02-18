"""Tests for priors with wrapped bounds."""

from sampling.priors import WrappedPrior


def test_wrapped_prior_has_wrap_bounds() -> None:
    """Test that WrappedPrior has wrap bounds."""

    prior = WrappedPrior(
        wrap_bounds=[(-180.0, 180.0)],
    )
    assert prior.wrap_bounds == [(-180.0, 180.0)]
