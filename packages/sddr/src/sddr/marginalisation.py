"""Marginalise Priors."""

from collections.abc import Sequence
from functools import singledispatch
from typing import overload

import numpy as np
from sampling.priors._protocols import PriorFunction
from sampling.priors._utils import _normalise_indices
from sampling.priors.compound import CompoundPrior, PriorComponent
from sampling.priors.gaussian import GaussianPrior
from sampling.priors.uniform import UniformPrior


@overload
def marginalise_prior(
    prior: GaussianPrior, indices: Sequence[int] | slice | np.ndarray
) -> GaussianPrior: ...


@overload
def marginalise_prior(
    prior: UniformPrior, indices: Sequence[int] | slice | np.ndarray
) -> UniformPrior: ...


@overload
def marginalise_prior(
    prior: CompoundPrior, indices: Sequence[int] | slice | np.ndarray
) -> CompoundPrior: ...


@overload
def marginalise_prior(
    prior: PriorFunction, indices: Sequence[int] | slice | np.ndarray
) -> PriorFunction: ...


@singledispatch
def marginalise_prior(
    prior: PriorFunction, indices: Sequence[int] | slice | np.ndarray
) -> PriorFunction:
    r"""
    Create a marginalised prior function over specified parameter indices.

    This assumes that all the random variables are independent.

    .. math::
        p(m_{\text{remaining}}) = \int p(m_{\text{remaining}}, m_{\text{marginalised}}) \, dm_{\text{marginalised}}
        = \int p(m_{\text{remaining}}) p(m_{\text{marginalised}}) \, dm_{\text{marginalised}}
        = p(m_{\text{remaining}}) \int p(m_{\text{marginalised}}) \, dm_{\text{marginalised}}
        = p(m_{\text{remaining}})

    Parameters
    ----------
    prior : PriorFunction
        Original prior function to marginalise.
    indices : Sequence[int] | slice | np.ndarray
        Indices of the model parameters to keep after marginalisation i.e. $m_{\text{remaining}}$.

    Returns
    -------
    marginal_prior_fn : PriorFunction
        Marginalised prior function that takes model parameters and returns the log-prior.

    Raises
    ------
    ValueError
        If no indices are provided (i.e., the indices array is empty).
    IndexError
        If any of the provided indices are out of bounds for the parameter array.
    """
    idx = _normalise_indices(indices, prior.n)
    if idx.size == 0:
        raise ValueError(
            "At least one index should be kept after marginalisation.  Check the 'indices' parameter."
        )

    marginalised_params: list[np.ndarray] = []
    for param in prior.config_params:
        # repeat idx for each dimension of param
        # e.g. for a covariance matrix, we need to select both rows and columns
        ndim_repeat = tuple([idx] * param.ndim)
        # create the n-dimensional index selector
        selector = np.ix_(*ndim_repeat)
        marginalised_params.append(param[selector])

    return type(prior)(*marginalised_params)


@marginalise_prior.register
def _(
    compound_prior: CompoundPrior, indices: Sequence[int] | slice | np.ndarray
) -> CompoundPrior:
    """
    Create a marginalised compound prior function over specified parameter indices.

    This assumes that all the random variables are independent.

    Parameters
    ----------
    compound_prior : CompoundPrior
        Original compound prior function to marginalise.
    indices : Sequence[int] | slice | np.ndarray
        Indices of the model parameters to keep after marginalisation.

    Returns
    -------
    marginal_compound_prior : CompoundPrior
        Marginalised compound prior function that takes model parameters and returns the log-prior.

    Raises
    ------
    ValueError
        If no indices are provided to keep after marginalisation, or if no prior components remain after marginalisation.
    """
    idx = _normalise_indices(indices, compound_prior.n)
    if idx.size == 0:
        raise ValueError(
            "At least one index should be kept after marginalisation. Check the 'indices' parameter."
        )

    # For each component, find which of the requested indices belong to it
    new_components = []
    next_index = 0  # Track the starting position in the new compound space

    for component in compound_prior.prior_components:
        component_indices = np.asarray(component.indices)

        # Find which requested indices belong to this component
        mask = np.isin(idx, component_indices)
        if not np.any(mask):
            continue  # No indices from this component are being kept

        kept_global_indices = idx[mask]

        # Map global indices to local indices within the component
        local_indices = np.array(
            np.searchsorted(component_indices, kept_global_indices)
        )

        # Marginalise the prior function to just these parameters
        marginalised_prior = marginalise_prior(component.prior_fn, local_indices)

        # Create new component with indices in the new compound space
        n_kept = local_indices.size
        new_component = PriorComponent(
            prior_fn=marginalised_prior,
            indices=np.arange(next_index, next_index + n_kept),
        )
        new_components.append(new_component)
        next_index += n_kept

    if not new_components:
        raise ValueError(
            "No prior components remain after marginalisation. Check the 'indices' parameter."
        )

    return CompoundPrior(new_components)


def marginalise_samples(
    samples: np.ndarray, param_indices: Sequence[int] | slice
) -> np.ndarray:
    """
    Extract marginal posterior samples for specified parameter indices.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples.
    param_indices : Sequence[int] | slice
        Indices of parameters to extract (i.e., columns to keep).

    Returns
    -------
    marginal_samples : ndarray, shape (n_samples, n_selected_params)
        Posterior samples for the selected parameters.
    """
    return samples[:, param_indices]
