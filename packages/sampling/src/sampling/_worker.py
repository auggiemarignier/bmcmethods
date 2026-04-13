"""Globals and worker initialisers for multiprocessing."""

import os
import warnings
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Self

import numpy as np

from ._util import DummyPool
from .likelihood._base import ForwardBase, ForwardGradientBase, LikelihoodBase
from .priors import PriorFunction

WORKER_FORWARD: None | ForwardBase = None
WORKER_FORWARD_GRADIENT: None | ForwardGradientBase = None
WORKER_LIKELIHOOD: None | LikelihoodBase = None
WORKER_PRIOR: None | PriorFunction = None


@dataclass(frozen=True)
class _IdentityState:
    pass


class _IdentityForward(ForwardBase[_IdentityState]):
    """Internal no-op for when no forward model is provided.

    Assumes model parameters are in data-space.
    """

    state = _IdentityState()

    @classmethod
    def from_state(cls, state: _IdentityState) -> Self:
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        return model_params


class _NoForwardGradient(ForwardGradientBase[_IdentityState]):
    """Internal no-op for when no forward model gradient is provided.

    Will raise an error if something tries to call this.
    """

    state = _IdentityState()

    @classmethod
    def from_state(cls, state: _IdentityState) -> Self:
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        raise RuntimeError("A gradient for the forward function is needed.")


@dataclass(frozen=True)
class _MCMCSpec:
    """Specification of MCMC objects."""

    likelihood_cls: type[LikelihoodBase[Any]]
    likelihood_state: Any
    prior: PriorFunction
    forward_cls: type[ForwardBase[Any]]
    forward_state: Any
    forward_gradient_cls: type[ForwardGradientBase[Any]]
    forward_gradient_state: Any

    @classmethod
    def from_likelihood_and_prior(
        cls, likelihood: LikelihoodBase[Any], prior: PriorFunction
    ) -> Self:
        fwd = (
            likelihood.forward if likelihood.forward is not None else _IdentityForward()
        )
        fwd_grad = (
            likelihood.forward_gradient
            if likelihood.forward_gradient is not None
            else _NoForwardGradient()
        )

        return cls(
            likelihood_cls=likelihood.__class__,
            likelihood_state=likelihood.state,
            prior=prior,
            forward_cls=fwd.__class__,
            forward_state=fwd.state,
            forward_gradient_cls=fwd_grad.__class__,
            forward_gradient_state=fwd_grad.state,
        )


def make_pool(parallel: bool | int, likelihood: LikelihoodBase, prior: PriorFunction):
    """Create a pool object, initialising each worker with heavy read-only data to avoid large copies at each iteration."""
    spec = _MCMCSpec.from_likelihood_and_prior(likelihood, prior)
    args = (  # could use astuple but just being explicit
        spec.likelihood_cls,
        spec.likelihood_state,
        spec.prior,
        spec.forward_cls,
        spec.forward_state,
        spec.forward_gradient_cls,
        spec.forward_gradient_state,
    )

    if not parallel:
        init_worker(*args)
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
        initargs=args,
    )

    return pool


def init_worker[FS, LS, FGS](
    likelihood_cls: type[LikelihoodBase[LS]],
    likelihood_state: LS,
    prior: PriorFunction,
    forward_cls: type[ForwardBase[FS]],
    forward_state: FS,
    forward_gradient_cls: type[ForwardGradientBase[FGS]],
    forward_gradient_state: FGS,
) -> None:
    """
    Runs once in each worker process.

    Rebuilds the likelihood locally from the supplied state.
    """
    global WORKER_FORWARD, WORKER_LIKELIHOOD, WORKER_PRIOR, WORKER_FORWARD_GRADIENT

    WORKER_FORWARD = forward_cls.from_state(forward_state)
    WORKER_FORWARD_GRADIENT = forward_gradient_cls.from_state(forward_gradient_state)
    WORKER_LIKELIHOOD = likelihood_cls.from_state(
        likelihood_state,
        forward=WORKER_FORWARD,
        forward_gradient=WORKER_FORWARD_GRADIENT,
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
