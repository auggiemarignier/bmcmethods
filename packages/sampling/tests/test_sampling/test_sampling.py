"""Tests for the mcmc function in sampling module.

Focus: shape handling (per-walker burn-in and thinning), parallel flag
behaviour, and error on unimplemented prior initialisation.
"""

import numpy as np
from sampling.likelihood import GaussianLikelihood
from sampling.priors import UniformPrior
from sampling.sampling import MCMCConfig, mcmc, nuts


def likelihood_fn(theta: np.ndarray) -> np.ndarray:
    """Simple log-probability: standard normal (up to additive constant).

    Not in a fixture because needs to be top-level for pickling in multiprocessing.
    Supports both scalar (1D) and batched (2D) inputs.
    """
    theta = np.atleast_2d(theta)
    return -0.5 * np.sum(theta * theta, axis=1).squeeze()


class LogPrior:
    """Dummy prior for testing.

    Not in a fixture because needs to be top-level for pickling in multiprocessing.
    Supports both scalar (1D) and batched (2D) inputs.
    """

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        model_params = np.atleast_2d(model_params)
        return np.zeros(model_params.shape[0]).squeeze()

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from standard normal prior."""
        return rng.normal(0, 1, size=(num_samples, self.n))

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        """Gradient of the log-prior (zero for standard normal)."""
        model_params = np.atleast_2d(model_params)
        return np.zeros_like(model_params)

    @property
    def n(self) -> int:
        """Number of parameters."""
        return 2

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return []


def test_mcmc_shapes_no_thin(
    rng: np.random.Generator,
) -> None:
    """Validate returned sample and log-probability shapes without thinning."""
    prior = LogPrior()
    ndim = prior.n
    cfg = MCMCConfig(nwalkers=6, nsteps=6, burn_in=2, thin=1, parallel=False)
    samples, lnprob = mcmc(ndim, likelihood_fn, prior, rng, cfg)
    expected_n = (cfg.nsteps - cfg.burn_in) * cfg.nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)
    assert samples.flags.c_contiguous
    assert lnprob.flags.c_contiguous


def test_mcmc_shapes_with_thinning(
    rng: np.random.Generator,
) -> None:
    """Validate thinning reduces number of returned samples appropriately."""
    prior = LogPrior()
    ndim = prior.n
    cfg = MCMCConfig(nwalkers=5, nsteps=7, burn_in=3, thin=2, parallel=False)
    samples, lnprob = mcmc(ndim, likelihood_fn, prior, rng, cfg)
    steps_retained = cfg.nsteps - cfg.burn_in
    thinned_steps = (steps_retained + cfg.thin - 1) // cfg.thin
    expected_n = thinned_steps * cfg.nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)


def test_mcmc_parallel_flag(rng: np.random.Generator) -> None:
    """Run with parallel=True to ensure code path executes without error."""

    prior = LogPrior()
    ndim = prior.n
    cfg = MCMCConfig(nwalkers=4, nsteps=5, burn_in=1, thin=1, parallel=True)
    samples, lnprob = mcmc(ndim, likelihood_fn, prior, rng, cfg)
    expected_n = (cfg.nsteps - cfg.burn_in) * cfg.nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)


def test_mcmc_default_config(
    rng: np.random.Generator,
) -> None:
    """Run mcmc with default configuration to ensure no errors."""
    prior = LogPrior()
    ndim = prior.n
    samples, lnprob = mcmc(ndim, likelihood_fn, prior, rng)
    expected_n = (1000 - 200) * 50  # Default nsteps, burn_in, nwalkers
    assert samples.shape == (expected_n, ndim)
    assert lnprob.shape == (expected_n,)


def test_mcmc_excessive_burn_in_returns_full_chain(
    rng: np.random.Generator,
) -> None:
    """Excessive burn_in >= nsteps: burn-in ignored, full chain retained."""
    prior = LogPrior()
    ndim = prior.n
    cfg = MCMCConfig(nwalkers=4, nsteps=5, burn_in=10, thin=1, parallel=False)
    samples, lnprob = mcmc(ndim, likelihood_fn, prior, rng, cfg)
    assert samples.shape == (cfg.nsteps * cfg.nwalkers, ndim)
    assert lnprob.shape == (cfg.nsteps * cfg.nwalkers,)


def test_mcmc_excessive_thin_returns_unthinned(
    rng: np.random.Generator,
) -> None:
    """Thinning factor too large: thinning ignored after burn-in cut."""
    prior = LogPrior()
    ndim = prior.n
    cfg = MCMCConfig(nwalkers=4, nsteps=6, burn_in=2, thin=100, parallel=False)
    samples, lnprob = mcmc(ndim, likelihood_fn, prior, rng, cfg)
    expected_steps = cfg.nsteps - cfg.burn_in
    assert samples.shape == (expected_steps * cfg.nwalkers, ndim)
    assert lnprob.shape == (expected_steps * cfg.nwalkers,)


def test_nuts_uses_controller_and_returns_chains_and_logpdfs() -> None:
    ndim = 2
    nwalkers = 3
    nsteps = 4

    # Build a real GaussianLikelihood and a Prior matching PriorFunction (UniformPrior)
    # Forward function: identity mapping from model params to observed data
    def forward_fn(model_params: np.ndarray) -> np.ndarray:
        model_params = np.atleast_2d(model_params)
        return model_params  # (batch, ndim)

    def forward_fn_gradient(model_params: np.ndarray) -> np.ndarray:
        # Jacobian of identity is identity for each batch element
        model_params = np.atleast_2d(model_params)
        batch = model_params.shape[0]
        return np.tile(np.eye(ndim)[None, :, :], (batch, 1, 1))  # (batch, n_obs, ndim)

    observed = np.zeros(ndim)
    inv_covar = np.eye(ndim)

    like = GaussianLikelihood(
        forward_fn,
        observed,
        inv_covar,
        validate_covariance=True,
        example_model=np.zeros(ndim),
        forward_fn_gradient=forward_fn_gradient,
    )

    prior = UniformPrior(
        lower_bounds=np.full(ndim, -1.0), upper_bounds=np.full(ndim, 1.0)
    )

    rng = np.random.default_rng(42)

    # Call nuts
    cfg = MCMCConfig(nwalkers=nwalkers, nsteps=nsteps, progress=False)
    chains, log_pdf = nuts(ndim, like, prior, rng, config=cfg)

    # Assertions: shapes and simple content checks
    assert isinstance(chains, np.ndarray)
    assert isinstance(log_pdf, np.ndarray)
    assert chains.shape == (nsteps, nwalkers, ndim)
    assert log_pdf.shape == (nsteps, nwalkers)
