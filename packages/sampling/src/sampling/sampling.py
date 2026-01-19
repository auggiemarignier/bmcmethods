"""Sampling using emcee."""

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from emcee import EnsembleSampler

from ._util import DummyPool
from .posterior import Posterior
from .priors import PriorFunction


@dataclass(frozen=True)
class MCMCConfig:
    """Configuration for MCMC sampling.

    Parameters
    ----------
    nwalkers : int
        Number of MCMC walkers.
    nsteps : int
        Number of MCMC steps.
    burn_in : int
        Number of burn-in steps to discard.
    vectorise : bool
        Whether to vectorize the likelihood and prior evaluations.
    parallel : bool
        Whether to use parallel processing. Ignored if vectorise is True.
    progress : bool
        Whether to display a progress bar.
    thin : int
        Thinning factor for the MCMC samples.
    """

    nwalkers: int = 50
    nsteps: int = 1000
    burn_in: int = 200
    vectorise: bool = True
    parallel: bool = False
    progress: bool = False
    thin: int = 1


def mcmc(
    ndim: int,
    likelihood: Callable[[np.ndarray], float | np.ndarray],
    prior: PriorFunction,
    rng: np.random.Generator,
    config: MCMCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MCMC sampling using the ensemble sampler.

    Parameters
    ----------
    ndim : int
        Number of dimensions in the parameter space.
    likelihood : Callable[[ndarray], float | ndarray]
        Likelihood function that takes model parameters and returns log-likelihood.
        Should support both scalar (1D) and vectorised (2D batch) inputs if
        config.vectorise is True.
    prior : PriorFunction
        Prior function that takes model parameters and returns log-prior.
        Should support both scalar (1D) and vectorised (2D batch) inputs if
        config.vectorise is True.
    rng : np.random.Generator
        Random number generator for initializing walkers.
    config : MCMCConfig or None, optional
        MCMC configuration. If None, uses default configuration.

    Returns
    -------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters, after burn-in and thinning.
    lnprob : ndarray, shape (num_samples,)
        Log-probabilities of the MCMC samples, after burn-in and thinning.
    """
    if config is None:
        config = MCMCConfig()

    initial_pos = prior.sample(config.nwalkers, rng)

    posterior = Posterior(likelihood, prior)

    _pool = DummyPool
    if config.parallel and not config.vectorise:
        _pool = Pool

    with _pool() as pool:
        sampler = EnsembleSampler(
            config.nwalkers, ndim, posterior, pool=pool, vectorize=config.vectorise
        )
        sampler.run_mcmc(initial_pos, config.nsteps, progress=config.progress)

    return _burn_and_thin_sampler(sampler, config.burn_in, config.thin)


def _burn_and_thin_sampler(
    sampler: EnsembleSampler, burn_in: int, thin: int
) -> tuple[np.ndarray, np.ndarray]:
    """Apply burn-in and thinning to an MCMC chain from emcee.

    Parameters
    ----------
    sampler : EnsembleSampler
        MCMC sampler from emcee.

    Returns
    -------
    processed_chain : ndarray
        Processed MCMC chain after burn-in and thinning.
    processed_lnprob : ndarray
        Processed log-probabilities after burn-in and thinning.
    """
    chain = sampler.get_chain()  # shape (nsteps, nwalkers, ndim)
    chain = _burn_and_thin_array(chain, burn_in, thin)
    samples = np.ascontiguousarray(chain.reshape(-1, sampler.ndim))

    lnprob_chain = sampler.get_log_prob()  # shape (nsteps, nwalkers)
    lnprob_chain = _burn_and_thin_array(lnprob_chain, burn_in, thin)
    lnprob = np.ascontiguousarray(lnprob_chain.reshape(-1))

    return samples, lnprob


def _burn_and_thin_array(chain: np.ndarray, burn_in: int, thin: int) -> np.ndarray:
    """Apply burn-in and thinning to an MCMC chain.

    Parameters
    ----------
    chain : ndarray, shape (nsteps, nwalkers, ndim) or (nsteps, nwalkers)
        MCMC chain to process.
    burn_in : int
        Number of burn-in steps to discard.
    thin : int
        Thinning factor.

    Returns
    -------
    processed_chain : ndarray
        Processed MCMC chain after burn-in and thinning.
    """
    nd = chain.ndim
    if nd not in (2, 3):
        raise ValueError("Chain must be 2D or 3D ndarray.")

    if nd == 2:
        # Add dummy ndim axis for uniform processing
        chain = chain[:, :, np.newaxis]

    total_steps = chain.shape[0]
    burn_in_eff = burn_in if burn_in < total_steps else 0
    steps_after_burn = total_steps - burn_in_eff
    thin_eff = thin if (thin > 0 and steps_after_burn // thin > 0) else 1

    processed_chain = chain[burn_in_eff::thin_eff, :, :]
    if nd == 2:
        # Remove dummy ndim axis if it was added
        processed_chain = processed_chain[:, :, 0]

    return processed_chain
