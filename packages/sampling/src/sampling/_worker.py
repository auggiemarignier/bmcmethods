"""Globals and worker initialisers for multiprocessing."""

import os
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Self

import numpy as np

from ._util import DummyPool
from .likelihood._base import ForwardBase, ForwardGradientBase, LikelihoodBase
from .priors import PriorFunction

WORKER_FORWARD: None | ForwardBase = None
WORKER_FORWARD_GRADIENT: None | ForwardGradientBase = None
WORKER_LIKELIHOOD: None | LikelihoodBase = None
WORKER_PRIOR: None | PriorFunction = None


@dataclass(frozen=True)
class _MCMCSpec[FS, LS, FGS]:
    """Specification of MCMC objects."""

    likelihood_cls: type[LikelihoodBase[LS]]
    likelihood_state: LS
    prior: PriorFunction
    forward_cls: type[ForwardBase[FS]] | None = None
    forward_state: FS | None = None
    forward_gradient_cls: type[ForwardGradientBase[FGS]] | None = None
    forward_gradient_state: FGS | None = None

    @classmethod
    def from_likelihood_and_prior(
        cls, likelihood: LikelihoodBase[LS], prior: PriorFunction
    ) -> Self:
        fwd = likelihood.forward
        forward_cls = None if fwd is None else fwd.__class__
        forward_state = None if fwd is None else fwd.state

        fwd_grad = likelihood.forward_gradient
        forward_grad_cls = None if fwd_grad is None else fwd_grad.__class__
        forward_grad_state = None if fwd_grad is None else fwd_grad.state

        return cls(
            likelihood_cls=likelihood.__class__,
            likelihood_state=likelihood.state,
            prior=prior,
            forward_cls=forward_cls,
            forward_state=forward_state,
            forward_gradient_cls=forward_grad_cls,
            forward_gradient_state=forward_grad_state,
        )


def make_pool(parallel: bool | int, likelihood: LikelihoodBase, prior: PriorFunction):
    """Create a pool object, initialising each worker with heavy read-only data to avoid large copies at each iteration."""
    spec = _MCMCSpec.from_likelihood_and_prior(likelihood, prior)

    if not parallel:
        init_worker(
            spec.forward_cls,
            spec.forward_state,
            spec.likelihood_cls,
            spec.likelihood_state,
            spec.prior,
            spec.forward_gradient_cls,
            spec.forward_gradient_state,
        )
        return DummyPool()

    if isinstance(parallel, bool):
        processes = os.cpu_count()
        if processes is None:
            warnings.warn(
                "Could not determine CPU count; falling back to 1 process.",
                RuntimeWarning,
                stacklevel=2,
            )
            processes = 1
    elif isinstance(parallel, int):
        processes = parallel
    else:
        raise TypeError("Invalid type for config.parallel. Must be bool or int.")

    pool = Pool(
        processes=processes,
        initializer=init_worker,
        initargs=(
            spec.forward_cls,
            spec.forward_state,
            spec.likelihood_cls,
            spec.likelihood_state,
            spec.prior,
            spec.forward_gradient_cls,
            spec.forward_gradient_state,
        ),
    )

    return pool


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
