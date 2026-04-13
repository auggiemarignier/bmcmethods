"""Sampling using emcee."""

from dataclasses import dataclass

import numpy as np
from emcee import EnsembleSampler
from pints import LogPDF, MCMCController, NoUTurnMCMC
from ptemcee import Sampler
from tqdm import tqdm

from sampling._worker import logl, logp, make_pool
from sampling.likelihood._base import LikelihoodBase

from .likelihood import GaussianLikelihood
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
    parallel : bool or int
        Whether to use parallel processing. If an integer is given, it specifies the number of processes to use.
    progress : bool
        Whether to display a progress bar.
    """

    nwalkers: int = 50
    nsteps: int = 1000
    parallel: bool | int = True
    progress: bool = True


def mcmc(
    ndim: int,
    likelihood: LikelihoodBase,
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
        Should support both scalar (1D) and vectorised (2D batch) inputs.
    prior : PriorFunction
        Prior function that takes model parameters and returns log-prior.
        Should support both scalar (1D) and vectorised (2D batch) inputs.
    rng : np.random.Generator
        Random number generator for initializing walkers.
    config : MCMCConfig or None, optional
        MCMC configuration. If None, uses default configuration.

    Returns
    -------
    samples : ndarray, shape (nsteps, nwalkers, ndim)
        MCMC samples of the model parameters.
    lnprob : ndarray, shape (nsteps, nwalkers,)
        Log-probabilities of the MCMC samples.
    """
    if config is None:
        config = MCMCConfig()

    pool_cm = make_pool(config.parallel, likelihood, prior)
    posterior = Posterior(logl, logp)
    initial_pos = prior.sample(config.nwalkers, rng)

    with pool_cm as pool:
        sampler = EnsembleSampler(config.nwalkers, ndim, posterior, pool=pool)
        sampler.run_mcmc(initial_pos, config.nsteps, progress=config.progress)

    return sampler.get_chain(), sampler.get_log_prob()


def ptmcmc[S](
    ndim: int,
    likelihood: LikelihoodBase[S],
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
        Should support both scalar (1D) and vectorised (2D batch) inputs.
    prior : PriorFunction
        Prior function that takes model parameters and returns log-prior.
        Should support both scalar (1D) and vectorised (2D batch) inputs.
    rng : np.random.Generator
        Random number generator for initializing walkers.
    config : MCMCConfig or None, optional
        MCMC configuration. If None, uses default configuration.

    Returns
    -------
    samples : ndarray, shape (ntemps (10), nsteps, nwalkers, ndim)
        MCMC samples of the model parameters.
    lnprob : ndarray, shape (ntemps (10), nsteps, nwalkers)
        Log-probabilities of the MCMC samples.
    """
    if config is None:
        config = MCMCConfig()

    ntemps = 10

    initial_pos = prior.sample(ntemps * config.nwalkers, rng).reshape(
        (ntemps, config.nwalkers, ndim)
    )

    pool_cm = make_pool(config, likelihood.__class__, likelihood.state, prior)

    with pool_cm as pool:
        sampler = Sampler(
            config.nwalkers,
            ndim,
            logl,
            logp,
            pool=pool,
            ntemps=ntemps,
        )
        for _ in tqdm(
            sampler.sample(initial_pos, config.nsteps),
            total=config.nsteps,
            disable=not config.progress,
        ):
            pass

    return sampler.chain.swapaxes(1, 2), sampler.logprobability.swapaxes(1, 2)


class PintsPDF(LogPDF):
    """Wrapper to use our Posterior with pints."""

    def __init__(self, posterior: Posterior, ndim: int):
        self.posterior = posterior
        self.ndim = ndim

    def __call__(self, x):
        """Evaluate the log-posterior at given model parameters."""
        return self.posterior(x)

    def evaluateS1(self, x):
        """Evaluate the log-posterior and its gradient at given model parameters."""
        return self.posterior(x), self.posterior.gradient(x)

    def n_parameters(self):
        """Return the number of parameters in the model."""
        return self.ndim


def nuts(
    ndim: int,
    likelihood: GaussianLikelihood,
    prior: PriorFunction,
    rng: np.random.Generator,
    config: MCMCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """NUTS sampling using pints.

    Parameters
    ----------
    ndim : int
        Number of dimensions in the parameter space.
    likelihood : GaussianLikelihood
        Gaussian likelihood function with gradient support.
    prior : PriorFunction
        Prior function with gradient support.
    rng : np.random.Generator
        Random number generator for initializing walkers.
    config : MCMCConfig or None, optional
        MCMC configuration. If None, uses default configuration.
        The following parameters are used: ``nwalkers``, ``nsteps``,
        ``progress``, ``parallel``.

    Returns
    -------
    samples : ndarray, shape (nsteps, nwalkers, ndim)
        MCMC samples of the model parameters for the full chain.
    lnprob : ndarray, shape (nsteps, nwalkers,)
        Log-probabilities of the MCMC samples for the full chain.
    """
    # TODO: Sort out how the gradients data gets passed to the workers

    if config is None:
        config = MCMCConfig()

    posterior = Posterior(likelihood, prior, likelihood.gradient, prior.gradient)

    initial_pos = prior.sample(config.nwalkers, rng)
    nuts_mcmc = MCMCController(
        PintsPDF(posterior, ndim), config.nwalkers, initial_pos, method=NoUTurnMCMC
    )
    nuts_mcmc.set_max_iterations(config.nsteps)
    nuts_mcmc.set_log_to_screen(config.progress)
    nuts_mcmc.set_log_pdf_storage(True)
    nuts_mcmc.set_parallel(config.parallel)

    chains = nuts_mcmc.run()
    log_pdf = nuts_mcmc.log_pdfs()

    return chains.swapaxes(0, 1), log_pdf.swapaxes(0, 1)
