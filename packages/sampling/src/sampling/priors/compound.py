"""Compound Prior combining multiple prior components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from itertools import chain
from typing import Any

import numpy as np

from ._protocols import PriorComponentConfig, PriorType
from .component import PriorComponent
from .gaussian import GaussianPriorComponentConfig
from .uniform import UniformPrior, UniformPriorComponentConfig

_CONFIG_FACTORIES: dict[PriorType, type[PriorComponentConfig]] = {
    PriorType.GAUSSIAN: GaussianPriorComponentConfig,
    PriorType.UNIFORM: UniformPriorComponentConfig,
}


class CompoundPrior:
    """
    Represents a compound prior formed by combining multiple prior components.

    A compound prior is a joint prior distribution over model parameters, constructed
    by combining several prior components, each of which acts on a subset of the parameters.
    This is useful when different groups of parameters have different prior distributions,
    or when the overall prior can be factorized into independent components.

    When evaluating the compound prior, any UniformPrior components are reordered to be
    evaluated first. This allows for early exit optimization: if any UniformPrior component
    returns -inf (indicating the parameters are outside the allowed range), the evaluation
    stops immediately and -inf is returned for the whole compound prior.

    Parameters
    ----------
    prior_components : Sequence[PriorComponent]
        Sequence of PriorComponent instances, each specifying a prior and the indices
        of the model parameters it applies to.
    vectorised : bool, optional
        If True, the prior can evaluate batches of models (shape (batch_size, n)).
        If False, evaluates single models (shape (n,)). Default is True.

    Raises
    ------
    IndexError
        If the indices specified in any PriorComponent are invalid for the given model parameters.
    TypeError
        If the input types for model parameters or prior components are incorrect.
    ValueError
        If the prior components do not cover the expected number of parameters, or if there is overlap.
    """

    def __init__(
        self, prior_components: Sequence[PriorComponent], vectorised: bool = False
    ) -> None:
        self.prior_components = prior_components
        self._n = sum(c.n for c in prior_components)

        self._uniform_components = [
            c for c in prior_components if isinstance(c.prior_fn, UniformPrior)
        ]
        self._non_uniform_components = [
            c for c in prior_components if not isinstance(c.prior_fn, UniformPrior)
        ]
        self._call_fn = self._call_vectorised if vectorised else self._call_single

    def __call__(self, model_params: np.ndarray) -> float | np.ndarray:
        """Compound log-prior.

        Parameters
        ----------
        model_params : ndarray
            If vectorised=False: shape (n,) for single model evaluation.
            If vectorised=True: shape (batch_size, n) for batch evaluation.

        Returns
        -------
        float or ndarray
            If vectorised=False: scalar log-prior value.
            If vectorised=True: array of shape (batch_size,) with log-prior values.
        """
        return self._call_fn(model_params)

    def _call_single(self, model_params: np.ndarray) -> float:
        """Compound log-prior for a single model."""
        # Bring any UniformPriors to the front for early exit
        prior_components = chain(self._uniform_components, self._non_uniform_components)

        total_log_prior = 0.0
        for component in prior_components:
            params_subset = model_params[component.indices]
            component_log_prior = component.prior_fn(params_subset)

            if np.isneginf(component_log_prior):
                return -np.inf  # Early exit if any component is -inf

            total_log_prior += component_log_prior
        return total_log_prior

    def _call_vectorised(self, model_params: np.ndarray) -> np.ndarray:
        """Compound log-prior for a batch of models.

        Parameters
        ----------
        model_params : ndarray, shape (batch_size, n)
            Batch of model parameters.

        Returns
        -------
        log_priors : ndarray, shape (batch_size,)
            Log-prior values for each model.
        """
        batch_size = model_params.shape[0]
        total_log_priors = np.zeros(batch_size)

        # Bring any UniformPriors to the front for early exit
        prior_components = chain(self._uniform_components, self._non_uniform_components)

        for component in prior_components:
            params_subset = model_params[:, component.indices]
            component_log_priors = component.prior_fn(params_subset)

            # Check for -inf values (out of bounds)
            invalid_mask = np.isneginf(component_log_priors)
            if np.any(invalid_mask):
                total_log_priors[invalid_mask] = -np.inf

            # Only add to valid entries
            valid_mask = ~invalid_mask
            total_log_priors[valid_mask] += component_log_priors[valid_mask]

        return total_log_priors

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Sample from the compound prior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (num_samples, n)
            Samples drawn from the compound prior.
        """
        samples = np.empty((num_samples, self._n))

        for component in self.prior_components:
            component_samples = component.prior_fn.sample(num_samples, rng)
            samples[:, component.indices] = component_samples

        return samples

    @property
    def n(self) -> int:
        """Total number of parameters in the compound prior."""
        return self._n

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CompoundPrior:
        """Build a CompoundPrior from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            Dictionary with a 'components' key containing a list of component configs.

        Returns
        -------
        CompoundPrior
            Compound prior built from the provided configuration.
        """
        config = CompoundPriorConfig.from_dict(config_dict)
        return config.to_compound_prior()


@dataclass
class CompoundPriorConfig:
    """Configuration for a compound prior with multiple components.

    Parameters
    ----------
    components : list[GaussianPriorConfig | UniformPriorConfig]
        List of prior component configurations.
    vectorised : bool, optional
        If True, the prior can evaluate batches of models. Default is True.

    Examples
    --------
    From a YAML file:

    .. code-block:: yaml

        components:
          - type: gaussian
            mean: [0.0, 0.0]
            inv_covar: [[1.0, 0.0], [0.0, 1.0]]
            indices: [0, 1]
          - type: uniform
            lower_bounds: [-1.0, -1.0]
            upper_bounds: [1.0, 1.0]
            indices: [2, 3]

    Load and build:

    >>> with open("prior_config.yaml") as f:
    ...     config_dict = yaml.safe_load(f)
    >>> config = CompoundPriorConfig.from_dict(config_dict)
    >>> prior = config.to_compound_prior()
    """

    components: list[PriorComponentConfig] = field(default_factory=list)
    vectorised: bool = True

    def __post_init__(self) -> None:
        """Validate component configurations."""
        for comp in self.components:
            try:
                PriorType(comp.type.lower())
            except ValueError as e:
                raise ValueError(f"Unknown prior type: {comp.type}") from e
            except AttributeError as e:
                raise AttributeError("Component configuration missing a 'type'.") from e

        total_indices = list(
            chain.from_iterable(comp.indices for comp in self.components)
        )
        n_params = len(total_indices)

        if sorted(total_indices) != list(range(n_params)):
            raise ValueError(
                "Prior components must cover all parameter indices without overlap."
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CompoundPriorConfig:
        """Build configuration from a dictionary (e.g., loaded from YAML).

        Parameters
        ----------
        config_dict : dict
            Dictionary with a 'components' key containing a list of component configs.
            Additional recognised keys (e.g., 'vectorised') are passed to the constructor.
            Unknown keys are silently ignored.

        Returns
        -------
        CompoundPriorConfig
            Configuration instance ready to build a CompoundPrior.
        """
        _config_dict = config_dict.copy()
        component_dicts = _config_dict.pop("components")

        component_configs = []
        for comp_dict in component_dicts:
            _comp_dict = comp_dict.copy()
            comp_type = _comp_dict.pop("type", None)
            if comp_type is None:
                raise ValueError("Each component config must have a 'type' key.")

            try:
                comp_type = PriorType(comp_type.lower())
            except ValueError as e:
                raise ValueError(f"Unknown prior type: {comp_type}") from e

            factory_cls = _CONFIG_FACTORIES.get(comp_type)
            if factory_cls is None:
                raise ValueError(f"Unknown prior type: {comp_type}")

            component_config = factory_cls(**_comp_dict)
            component_configs.append(component_config)

        # Only pass known dataclass fields, silently ignore unknown keys
        # Derive allowed keys from dataclass fields, excluding 'components'
        known_fields = {f.name for f in fields(cls) if f.name != "components"}
        config_kwargs = {k: v for k, v in _config_dict.items() if k in known_fields}

        return cls(components=component_configs, **config_kwargs)

    def to_compound_prior(self) -> CompoundPrior:
        """Build a CompoundPrior from this configuration.

        Returns
        -------
        CompoundPrior
            Compound prior built from all component configurations.
        """
        # Update component configs with the compound prior's vectorised setting
        for comp in self.components:
            comp.vectorised = self.vectorised

        prior_components = [comp.to_prior_component() for comp in self.components]
        # Only forward explicitly supported keyword arguments to CompoundPrior
        # Derive allowed keys from dataclass fields, excluding 'components'
        known_fields = {f.name for f in fields(self) if f.name != "components"}
        kwargs = {k: v for k, v in self.__dict__.items() if k in known_fields}
        return CompoundPrior(prior_components, **kwargs)
