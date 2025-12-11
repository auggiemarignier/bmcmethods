"""Tests for the mcmc function in sampling module.

Focus: shape handling (per-walker burn-in and thinning), parallel flag
behaviour, and error on unimplemented prior initialisation.
"""

import numpy as np
import pytest
from sampling.sampling import MCMCConfig, mcmc


def simple_log_p(theta: np.ndarray) -> float:
    """Simple log-probability: standard normal (up to additive constant)."""
    return -0.5 * float(theta @ theta)


def test_mcmc_shapes_no_thin(rng: np.random.Generator) -> None:
    """Validate returned sample and log-probability shapes without thinning."""
    ndim = 3
    # emcee requires nwalkers >= 2*ndim for the default stretch move
    cfg = MCMCConfig(nwalkers=6, nsteps=6, burn_in=2, thin=1, parallel=False)
    samples, lnprob = mcmc(ndim, simple_log_p, rng, cfg)
    # Expected length: (nsteps - burn_in) * nwalkers after per-walker burn-in.
    expected_n = (cfg.nsteps - cfg.burn_in) * cfg.nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)
    assert samples.flags.c_contiguous
    assert lnprob.flags.c_contiguous


def test_mcmc_shapes_with_thinning(rng: np.random.Generator) -> None:
    """Validate thinning reduces number of returned samples appropriately."""
    ndim = 2
    cfg = MCMCConfig(nwalkers=5, nsteps=7, burn_in=3, thin=2, parallel=False)
    samples, lnprob = mcmc(ndim, simple_log_p, rng, cfg)
    steps_retained = cfg.nsteps - cfg.burn_in
    thinned_steps = (steps_retained + cfg.thin - 1) // cfg.thin
    expected_n = thinned_steps * cfg.nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)


def test_mcmc_parallel_flag(rng: np.random.Generator) -> None:
    """Run with parallel=True to ensure code path executes without error."""
    ndim = 2
    cfg = MCMCConfig(nwalkers=4, nsteps=5, burn_in=1, thin=1, parallel=True)
    samples, lnprob = mcmc(ndim, simple_log_p, rng, cfg)
    expected_n = (cfg.nsteps - cfg.burn_in) * cfg.nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)


def test_mcmc_initial_from_prior_not_implemented(rng: np.random.Generator) -> None:
    """Setting initial_from_prior=True raises NotImplementedError."""
    ndim = 1
    cfg = MCMCConfig(
        nwalkers=3, nsteps=4, burn_in=1, thin=1, parallel=False, initial_from_prior=True
    )
    with pytest.raises(NotImplementedError):
        _ = mcmc(ndim, simple_log_p, rng, cfg)


def test_mcmc_default_config(rng: np.random.Generator) -> None:
    """Run mcmc with default configuration to ensure no errors."""
    ndim = 2
    samples, lnprob = mcmc(ndim, simple_log_p, rng)
    expected_n = (1000 - 200) * 50  # Default nsteps, burn_in, nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)


def test_mcmc_excessive_burn_in_returns_full_chain(rng: np.random.Generator) -> None:
    """Excessive burn_in >= nsteps: burn-in ignored, full chain retained."""
    ndim = 2
    cfg = MCMCConfig(nwalkers=4, nsteps=5, burn_in=10, thin=1, parallel=False)
    samples, lnprob = mcmc(ndim, simple_log_p, rng, cfg)
    assert samples.shape == (cfg.nsteps * cfg.nwalkers, ndim)
    assert lnprob.shape == (cfg.nsteps * cfg.nwalkers,)


def test_mcmc_excessive_thin_returns_unthinned(rng: np.random.Generator) -> None:
    """Thinning factor too large: thinning ignored after burn-in cut."""
    ndim = 1
    cfg = MCMCConfig(nwalkers=4, nsteps=6, burn_in=2, thin=100, parallel=False)
    samples, lnprob = mcmc(ndim, simple_log_p, rng, cfg)
    expected_steps = cfg.nsteps - cfg.burn_in
    assert samples.shape == (expected_steps * cfg.nwalkers, ndim)
    assert lnprob.shape == (expected_steps * cfg.nwalkers,)
