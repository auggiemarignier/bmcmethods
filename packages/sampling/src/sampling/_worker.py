"""Globals and worker initialisers for multiprocessing."""

import numpy as np

from .likelihood._base import LikelihoodBase
from .priors import PriorFunction

WORKER_LIKELIHOOD: None | LikelihoodBase = None
WORKER_PRIOR: None | PriorFunction = None


def init_worker[S](
    likelihood_cls: type[LikelihoodBase[S]], likelihood_state: S, prior: PriorFunction
) -> None:
    """
    Runs once in each worker process.

    Rebuilds the likelihood locally from the supplied state.
    """
    global WORKER_LIKELIHOOD, WORKER_PRIOR
    WORKER_LIKELIHOOD = likelihood_cls.from_state(likelihood_state)
    WORKER_PRIOR = prior


def logl(theta: np.ndarray) -> float | np.ndarray:
    if WORKER_LIKELIHOOD is None:
        raise RuntimeError(
            "Worker likelihood is not initialised; call init_worker in each worker first."
        )
    return WORKER_LIKELIHOOD(theta)


def logp(theta: np.ndarray) -> float | np.ndarray:
    if WORKER_PRIOR is None:
        raise RuntimeError(
            "Worker prior is not initialised; call init_worker in each worker first."
        )
    return WORKER_PRIOR(theta)
