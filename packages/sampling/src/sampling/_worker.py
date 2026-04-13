"""Globals and worker initialisers for multiprocessing."""

import numpy as np

from .likelihood._base import ForwardBase, ForwardGradientBase, LikelihoodBase
from .priors import PriorFunction

WORKER_FORWARD: None | ForwardBase = None
WORKER_FORWARD_GRADIENT: None | ForwardGradientBase = None
WORKER_LIKELIHOOD: None | LikelihoodBase = None
WORKER_PRIOR: None | PriorFunction = None


def init_worker[FS, LS, FGS](
    forward_cls: type[ForwardBase[FS]],
    forward_state: FS,
    likelihood_cls: type[LikelihoodBase[LS]],
    likelihood_state: LS,
    prior: PriorFunction,
    forward_gradient_cls: type[ForwardGradientBase[FGS]] | None,
    forward_gradient_state: FGS | None,
) -> None:
    """
    Runs once in each worker process.

    Rebuilds the likelihood locally from the supplied state.
    """
    global WORKER_FORWARD, WORKER_LIKELIHOOD, WORKER_PRIOR, WORKER_FORWARD_GRADIENT

    WORKER_FORWARD = forward_cls.from_state(forward_state)

    if forward_gradient_cls is not None:
        WORKER_FORWARD_GRADIENT = forward_gradient_cls.from_state(
            forward_gradient_state
        )
    else:
        # explicitly setting to None here just in case some other process has modified the global variable somewhere.
        WORKER_FORWARD_GRADIENT = None

    WORKER_LIKELIHOOD = likelihood_cls.from_state(
        likelihood_state,
        forward_fn=WORKER_FORWARD,
        forward_fn_gradient=WORKER_FORWARD_GRADIENT,
    )
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
