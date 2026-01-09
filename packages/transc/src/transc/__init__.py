"""TransConceptual sampling methods for estimating model evidence."""

from pytransc.analysis.visits import get_relative_marginal_likelihoods as _grml
from pytransc.analysis.visits import get_visits_to_states
from pytransc.samplers import run_ensemble_resampler
from pytransc.utils.types import FloatArray


def get_relative_marginal_likelihoods(*args, walker_average="mean") -> FloatArray:
    """Get relative marginal likelihoods of visited states.

    This is a wrapper around the `get_relative_marginal_likelihoods` function
    from the `pytransc.analysis.visits` module.
    """
    visits, _ = get_visits_to_states(*args)
    return _grml(visits[-1], walker_average=walker_average)


__all__ = [
    "run_ensemble_resampler",
    "get_visits_to_states",
    "get_relative_marginal_likelihoods",
]
