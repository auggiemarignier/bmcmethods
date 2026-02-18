"""Wrapping priors around specified bounds."""


class WrappedPrior:
    """A prior that wraps parameters around specified bounds."""

    def __init__(self, wrap_bounds: list[tuple[float, float]]) -> None:
        """
        Initialize the WrappedPrior.

        Parameters
        ----------
        wrap_bounds : list of tuples
            List of (lower_bound, upper_bound) for each parameter to be wrapped.
        """
        self.wrap_bounds = wrap_bounds
