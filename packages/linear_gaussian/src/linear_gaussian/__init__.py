"""Bayesian analysis of linear combinations of Gaussian random variables."""

from .lg import (
    GaussianComponent,
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
    calc_posterior_predictive_cov,
    calc_posterior_predictive_mean,
    clear_cache,
)

__all__ = [
    "GaussianComponent",
    "calc_log_evidence",
    "calc_posterior_cov",
    "calc_posterior_mean",
    "calc_posterior_predictive_mean",
    "calc_posterior_predictive_cov",
    "clear_cache",
]
