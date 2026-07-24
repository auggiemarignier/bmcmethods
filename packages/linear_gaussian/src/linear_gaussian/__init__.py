"""Bayesian analysis of linear combinations of Gaussian random variables."""

from .lg import GaussianComponent, calc_log_evidence, calc_posterior_cov, calc_posterior_mean

__all__ = [
    "GaussianComponent",
    "calc_log_evidence",
    "calc_posterior_cov",
    "calc_posterior_mean",
]
