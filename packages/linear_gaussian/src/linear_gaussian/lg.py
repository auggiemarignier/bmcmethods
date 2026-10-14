"""Compatibility exports for the linear-Gaussian implementation.

Public functions are implemented in :mod:`linear_gaussian.api`; preparation
and numerical precision routines live in their respective modules.
"""

from .api import (
    calc_log_evidence,
    calc_posterior_cov,
    calc_posterior_mean,
    calc_posterior_predictive_cov,
    calc_posterior_predictive_mean,
)
from .cache import clear_cache
from .models import GaussianComponent

__all__ = [
    "GaussianComponent",
    "calc_log_evidence",
    "calc_posterior_cov",
    "calc_posterior_mean",
    "calc_posterior_predictive_cov",
    "calc_posterior_predictive_mean",
    "clear_cache",
]
