"""Tests for the mcmc function in sampling module."""

from typing import Self

import numpy as np
import pytest
from sampling.likelihood import GaussianLikelihood
from sampling.likelihood._base import ForwardBase, ForwardGradientBase, LikelihoodBase
from sampling.priors import UniformPrior
from sampling.sampling import MCMCConfig, mcmc, nuts, ptmcmc


class DummyLikelihood(LikelihoodBase):
    """Simple picklable likelihood with an integer state multiplier."""

    def __init__(self, factor: int) -> None:
        self.state = factor

    @classmethod
    def from_state(
        cls,
        state: int,
        *,
        forward: ForwardBase | None = None,
        forward_gradient: ForwardGradientBase | None = None,
    ) -> Self:
        return cls(state)

    def __call__(self, model_params: np.ndarray) -> float:
        return float(np.sum(model_params) * self.state)


class DummyPrior:
    """Simple prior object implementing the PriorFunction protocol."""

    def __init__(self, offset: float = 0.0, n: int = 1) -> None:
        self.config_params: list[np.ndarray] = []
        self.n = n
        self.offset = offset

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(model_params) + self.offset)

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        return np.zeros_like(model_params)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(num_samples, self.n))


@pytest.fixture
def likelihood() -> DummyLikelihood:
    return DummyLikelihood(1)


@pytest.fixture
def prior() -> DummyPrior:
    return DummyPrior()


def test_mcmc_shapes_no_thin(
    likelihood: DummyLikelihood,
    prior: DummyPrior,
    rng: np.random.Generator,
) -> None:
    """Validate returned sample and log-probability shapes."""
    ndim = prior.n
    cfg = MCMCConfig(nwalkers=6, nsteps=7, parallel=False)
    samples, lnprob = mcmc(ndim, likelihood, prior, rng, cfg)
    assert samples.shape == (cfg.nsteps, cfg.nwalkers, ndim)
    assert lnprob.shape == (cfg.nsteps, cfg.nwalkers)
    assert samples.flags.c_contiguous
    assert lnprob.flags.c_contiguous


def test_mcmc_parallel_flag(
    likelihood: DummyLikelihood,
    prior: DummyPrior,
    rng: np.random.Generator,
) -> None:
    """Run with parallel=True to ensure code path executes without error."""

    ndim = prior.n
    cfg = MCMCConfig(nwalkers=4, nsteps=5, parallel=True)
    samples, lnprob = mcmc(ndim, likelihood, prior, rng, cfg)
    assert samples.shape == (cfg.nsteps, cfg.nwalkers, ndim)
    assert lnprob.shape == (cfg.nsteps, cfg.nwalkers)


def test_ptmcmc_parallel_flag(
    likelihood: DummyLikelihood,
    prior: DummyPrior,
    rng: np.random.Generator,
) -> None:
    """Run ptmcmc with parallel=True to ensure code path executes without error."""

    ndim = prior.n
    # ptemcee requires an even number of walkers and >= 2*ndim
    cfg = MCMCConfig(nwalkers=2 * ndim, nsteps=5, parallel=True)
    # We only check that the function runs and returns non-empty outputs
    chain, logprob = ptmcmc(ndim, likelihood, prior, rng, cfg)
    assert chain.shape == (10, cfg.nsteps, cfg.nwalkers, ndim)
    assert logprob.shape == (10, cfg.nsteps, cfg.nwalkers)


def test_mcmc_default_config(
    likelihood: DummyLikelihood,
    prior: DummyPrior,
    rng: np.random.Generator,
) -> None:
    """Run mcmc with default configuration to ensure no errors."""
    ndim = prior.n
    samples, lnprob = mcmc(ndim, likelihood, prior, rng)
    assert samples.shape == (1000, 50, ndim)
    assert lnprob.shape == (1000, 50)


####################################################################################################
# NUTS
####################################################################################################


class Identity(ForwardBase[None]):
    state = None

    @classmethod
    def from_state(cls, state: None) -> Self:
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        model_params = np.atleast_2d(model_params)
        return model_params  # (batch, ndim)


class IdentityGrad(ForwardGradientBase[None]):
    state = None

    @classmethod
    def from_state(cls, state: None) -> Self:
        return cls()

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        # Jacobian of identity is identity for each batch element
        model_params = np.atleast_2d(model_params)
        batch = model_params.shape[0]
        ndim = model_params.shape[1]
        return np.tile(np.eye(ndim)[None, :, :], (batch, 1, 1))  # (batch, n_obs, ndim)


def test_nuts_uses_controller_and_returns_chains_and_logpdfs() -> None:
    ndim = 2
    nwalkers = 3
    nsteps = 4

    # Build a real GaussianLikelihood and a Prior matching PriorFunction (UniformPrior)
    # Forward function: identity mapping from model params to observed data

    observed = np.zeros(ndim)
    inv_covar = np.eye(ndim)

    like = GaussianLikelihood(
        Identity(),
        observed,
        inv_covar,
        validate_covariance=True,
        example_model=np.zeros(ndim),
        forward_gradient=IdentityGrad(),
    )

    prior = UniformPrior(
        lower_bounds=np.full(ndim, -1.0), upper_bounds=np.full(ndim, 1.0)
    )

    rng = np.random.default_rng(42)

    # Call nuts
    cfg = MCMCConfig(nwalkers=nwalkers, nsteps=nsteps, progress=False)
    chains, log_pdf = nuts(ndim, like, prior, rng, config=cfg)

    assert isinstance(chains, np.ndarray)
    assert isinstance(log_pdf, np.ndarray)
    assert chains.shape == (nsteps, nwalkers, ndim)
    assert log_pdf.shape == (nsteps, nwalkers)
