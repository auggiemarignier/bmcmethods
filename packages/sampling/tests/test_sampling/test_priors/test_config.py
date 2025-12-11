"""Tests for prior configuration classes."""

import numpy as np
import pytest
from sampling.priors import PriorComponent
from sampling.priors.compound import CompoundPrior, CompoundPriorConfig
from sampling.priors.gaussian import GaussianPrior, GaussianPriorComponentConfig
from sampling.priors.uniform import UniformPrior, UniformPriorComponentConfig


class TestUniformPriorComponentConfig:
    """Tests for UniformPriorComponentConfig."""

    def test_init_with_lists(self) -> None:
        """Test initialization with lists."""
        config = UniformPriorComponentConfig(
            lower_bounds=[0.0, 1.0],
            upper_bounds=[2.0, 3.0],
            indices=[0, 1],
        )

        assert config.lower_bounds == [0.0, 1.0]
        assert config.upper_bounds == [2.0, 3.0]
        assert config.indices == [0, 1]
        assert config.type == "uniform"

    def test_init_with_arrays(self) -> None:
        """Test initialization with numpy arrays."""
        lower = np.array([0.0, 1.0])
        upper = np.array([2.0, 3.0])
        indices = [0, 1]

        config = UniformPriorComponentConfig(
            lower_bounds=lower,
            upper_bounds=upper,
            indices=indices,
        )

        assert np.array_equal(config.lower_bounds, lower)
        assert np.array_equal(config.upper_bounds, upper)
        assert config.indices == indices

    def test_to_prior_component_with_lists(self) -> None:
        """Test conversion to PriorComponent from lists."""
        config = UniformPriorComponentConfig(
            lower_bounds=[0.0, 1.0],
            upper_bounds=[2.0, 3.0],
            indices=[0, 1],
        )

        component = config.to_prior_component()

        assert isinstance(component, PriorComponent)
        assert isinstance(component.prior_fn, UniformPrior)
        assert component.n == 2
        np.testing.assert_array_equal(component.indices, np.array([0, 1]))
        np.testing.assert_array_equal(
            component.prior_fn.lower_bounds, np.array([0.0, 1.0])
        )
        np.testing.assert_array_equal(
            component.prior_fn.upper_bounds, np.array([2.0, 3.0])
        )

    def test_to_prior_component_with_arrays(self) -> None:
        """Test conversion to PriorComponent from arrays."""
        lower = np.array([-1.0, -2.0, -3.0])
        upper = np.array([1.0, 2.0, 3.0])

        config = UniformPriorComponentConfig(
            lower_bounds=lower,
            upper_bounds=upper,
            indices=[2, 3, 4],
        )

        component = config.to_prior_component()

        assert isinstance(component, PriorComponent)
        assert component.n == 3
        np.testing.assert_array_equal(component.indices, np.array([2, 3, 4]))

    def test_to_prior_component_functional(self) -> None:
        """Test that the generated prior component functions correctly."""
        config = UniformPriorComponentConfig(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
            indices=[0, 1],
        )

        component = config.to_prior_component()

        # In bounds
        params_in = np.array([0.5, 0.5])
        assert component.prior_fn(params_in) == 0.0

        # Out of bounds
        params_out = np.array([1.5, 0.5])
        assert component.prior_fn(params_out) == -np.inf

    def test_to_prior_component_invalid_bounds(self) -> None:
        """Test that invalid bounds raise ValueError when building component."""
        config = UniformPriorComponentConfig(
            lower_bounds=[1.0, 0.0],
            upper_bounds=[0.0, 1.0],  # First bound is invalid
            indices=[0, 1],
        )

        with pytest.raises(ValueError, match="lower bound must be less than"):
            config.to_prior_component()

    def test_single_parameter_config(self) -> None:
        """Test configuration for a single parameter."""
        config = UniformPriorComponentConfig(
            lower_bounds=[5.0],
            upper_bounds=[10.0],
            indices=[3],
        )

        component = config.to_prior_component()
        assert component.n == 1
        assert component.prior_fn(np.array([7.0])) == 0.0
        assert component.prior_fn(np.array([11.0])) == -np.inf


class TestGaussianPriorComponentConfig:
    """Tests for GaussianPriorComponentConfig."""

    def test_init_with_lists(self) -> None:
        """Test initialization with lists."""
        config = GaussianPriorComponentConfig(
            mean=[0.0, 1.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 1],
        )

        assert config.mean == [0.0, 1.0]
        assert config.inv_covar == [[1.0, 0.0], [0.0, 1.0]]
        assert config.indices == [0, 1]
        assert config.type == "gaussian"

    def test_init_with_arrays(self) -> None:
        """Test initialization with numpy arrays."""
        mean = np.array([0.0, 1.0])
        inv_covar = np.eye(2)
        indices = [0, 1]

        config = GaussianPriorComponentConfig(
            mean=mean,
            inv_covar=inv_covar,
            indices=indices,
        )

        assert np.array_equal(config.mean, mean)
        assert np.array_equal(config.inv_covar, inv_covar)
        assert config.indices == indices

    def test_to_prior_component_with_lists(self) -> None:
        """Test conversion to PriorComponent from lists."""
        config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 1],
        )

        component = config.to_prior_component()

        assert isinstance(component, PriorComponent)
        assert isinstance(component.prior_fn, GaussianPrior)
        assert component.n == 2
        np.testing.assert_array_equal(component.indices, np.array([0, 1]))
        np.testing.assert_array_equal(component.prior_fn.mean, np.array([0.0, 0.0]))
        np.testing.assert_array_equal(component.prior_fn.inv_covar, np.eye(2))

    def test_to_prior_component_with_arrays(self) -> None:
        """Test conversion to PriorComponent from arrays."""
        mean = np.array([1.0, 2.0, 3.0])
        inv_covar = np.diag([0.5, 1.0, 2.0])

        config = GaussianPriorComponentConfig(
            mean=mean,
            inv_covar=inv_covar,
            indices=[0, 1, 2],
        )

        component = config.to_prior_component()

        assert isinstance(component, PriorComponent)
        assert component.n == 3
        np.testing.assert_array_equal(component.indices, np.array([0, 1, 2]))

    def test_to_prior_component_functional(self) -> None:
        """Test that the generated prior component functions correctly."""
        config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 1],
        )

        component = config.to_prior_component()

        # At the mean
        assert component.prior_fn(np.array([0.0, 0.0])) == 0.0

        # Away from mean
        log_prior = component.prior_fn(np.array([1.0, 1.0]))
        assert log_prior < 0.0
        np.testing.assert_almost_equal(log_prior, -1.0)

    def test_to_prior_component_with_correlation(self) -> None:
        """Test configuration with correlated covariance matrix."""
        config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[2.0, 0.5], [0.5, 2.0]],
            indices=[0, 1],
        )

        component = config.to_prior_component()

        # Check it creates valid prior
        assert isinstance(component.prior_fn, GaussianPrior)
        log_prior = component.prior_fn(np.array([1.0, 1.0]))
        assert np.isfinite(log_prior)
        assert log_prior < 0.0

    def test_to_prior_component_invalid_non_symmetric(self) -> None:
        """Test that non-symmetric covariance raises ValueError."""
        config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.5], [0.3, 1.0]],  # Non-symmetric
            indices=[0, 1],
        )

        with pytest.raises(ValueError, match="symmetric"):
            config.to_prior_component()

    def test_to_prior_component_invalid_not_positive_definite(self) -> None:
        """Test that non-positive-definite covariance raises ValueError."""
        config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 2.0], [2.0, 1.0]],  # Negative eigenvalue
            indices=[0, 1],
        )

        with pytest.raises(ValueError, match="positive semidefinite"):
            config.to_prior_component()

    def test_single_parameter_config(self) -> None:
        """Test configuration for a single parameter."""
        config = GaussianPriorComponentConfig(
            mean=[5.0],
            inv_covar=[[2.0]],
            indices=[2],
        )

        component = config.to_prior_component()
        assert component.n == 1
        assert component.prior_fn(np.array([5.0])) == 0.0


class TestCompoundPriorConfig:
    """Tests for CompoundPriorConfig."""

    def test_init_empty(self) -> None:
        """Test initialization with no components."""
        config = CompoundPriorConfig()
        assert config.components == []

    def test_init_with_components(self) -> None:
        """Test initialization with component list."""
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[0.0],
            upper_bounds=[1.0],
            indices=[0],
        )
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0],
            inv_covar=[[1.0]],
            indices=[1],
        )

        config = CompoundPriorConfig(components=[uniform_config, gaussian_config])

        assert len(config.components) == 2
        assert config.components[0] is uniform_config
        assert config.components[1] is gaussian_config

    def test_from_dict_with_uniform_component(self) -> None:
        """Test building config from dictionary with uniform component."""
        config_dict = {
            "components": [
                {
                    "type": "uniform",
                    "lower_bounds": [0.0, 1.0],
                    "upper_bounds": [2.0, 3.0],
                    "indices": [0, 1],
                }
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)

        assert len(config.components) == 1
        assert isinstance(config.components[0], UniformPriorComponentConfig)
        assert config.components[0].type == "uniform"
        assert config.components[0].lower_bounds == [0.0, 1.0]
        assert config.components[0].upper_bounds == [2.0, 3.0]
        assert config.components[0].indices == [0, 1]

    def test_from_dict_with_gaussian_component(self) -> None:
        """Test building config from dictionary with gaussian component."""
        config_dict = {
            "components": [
                {
                    "type": "gaussian",
                    "mean": [0.0, 0.0],
                    "inv_covar": [[1.0, 0.0], [0.0, 1.0]],
                    "indices": [0, 1],
                }
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)

        assert len(config.components) == 1
        assert isinstance(config.components[0], GaussianPriorComponentConfig)
        assert config.components[0].type == "gaussian"
        assert config.components[0].mean == [0.0, 0.0]
        assert config.components[0].inv_covar == [[1.0, 0.0], [0.0, 1.0]]
        assert config.components[0].indices == [0, 1]

    def test_from_dict_with_multiple_components(self) -> None:
        """Test building config from dictionary with multiple components."""
        config_dict = {
            "components": [
                {
                    "type": "gaussian",
                    "mean": [0.0, 0.0],
                    "inv_covar": [[1.0, 0.0], [0.0, 1.0]],
                    "indices": [0, 1],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-1.0, -1.0],
                    "upper_bounds": [1.0, 1.0],
                    "indices": [2, 3],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-6.0, -1.0],
                    "upper_bounds": [1.0, 5.0],
                    "indices": [4, 5],
                },
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)

        assert len(config.components) == 3
        assert isinstance(config.components[0], GaussianPriorComponentConfig)
        assert isinstance(config.components[1], UniformPriorComponentConfig)
        assert isinstance(config.components[2], UniformPriorComponentConfig)

    def test_from_dict_missing_type(self) -> None:
        """Test that missing 'type' key raises ValueError."""
        config_dict = {
            "components": [
                {
                    "lower_bounds": [0.0],
                    "upper_bounds": [1.0],
                    "indices": [0],
                }
            ]
        }

        with pytest.raises(ValueError, match="must have a 'type' key"):
            CompoundPriorConfig.from_dict(config_dict)

    def test_from_dict_unknown_type(self) -> None:
        """Test that unknown prior type raises ValueError."""
        config_dict = {
            "components": [
                {
                    "type": "exponential",
                    "rate": 1.0,
                    "indices": [0],
                }
            ]
        }

        with pytest.raises(ValueError, match="Unknown prior type"):
            CompoundPriorConfig.from_dict(config_dict)

    def test_to_compound_prior_single_component(self) -> None:
        """Test conversion to CompoundPrior with single component."""
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[0.0, 0.0],
            upper_bounds=[1.0, 1.0],
            indices=[0, 1],
        )
        config = CompoundPriorConfig(components=[uniform_config])

        prior = config.to_compound_prior()

        assert isinstance(prior, CompoundPrior)
        assert prior.n == 2
        assert len(prior.prior_components) == 1

    def test_to_compound_prior_multiple_components(self) -> None:
        """Test conversion to CompoundPrior with multiple components."""
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 1],
        )
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[-1.0, -1.0],
            upper_bounds=[1.0, 1.0],
            indices=[2, 3],
        )
        config = CompoundPriorConfig(components=[gaussian_config, uniform_config])

        prior = config.to_compound_prior()

        assert isinstance(prior, CompoundPrior)
        assert prior.n == 4
        assert len(prior.prior_components) == 2

    def test_to_compound_prior_functional(self) -> None:
        """Test that the generated compound prior functions correctly."""
        config_dict = {
            "components": [
                {
                    "type": "gaussian",
                    "mean": [0.0, 0.0],
                    "inv_covar": [[1.0, 0.0], [0.0, 1.0]],
                    "indices": [0, 1],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-1.0, -1.0],
                    "upper_bounds": [1.0, 1.0],
                    "indices": [2, 3],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-6.0, -1.0],
                    "upper_bounds": [1.0, 5.0],
                    "indices": [4, 5],
                },
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)
        prior = config.to_compound_prior()

        # Valid parameters
        params_valid = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
        log_prior = prior(params_valid)
        assert log_prior == 0.0

        # Invalid parameters (outside uniform bounds)
        params_invalid = np.array(
            [0.0, 0.0, 2.0, 0.5]
        )  # notice that early exit is allowing an array of incorrect size
        log_prior = prior(params_invalid)
        assert log_prior == -np.inf

    def test_compound_prior_from_dict_convenience_method(self) -> None:
        """Test CompoundPrior.from_dict convenience method."""
        config_dict = {
            "components": [
                {
                    "type": "uniform",
                    "lower_bounds": [0.0],
                    "upper_bounds": [1.0],
                    "indices": [0],
                }
            ]
        }

        prior = CompoundPrior.from_dict(config_dict)

        assert isinstance(prior, CompoundPrior)
        assert prior.n == 1

    def test_from_dict_with_different_index_ranges(self) -> None:
        """Test configuration with non-contiguous indices."""
        config_dict = {
            "components": [
                {
                    "type": "uniform",
                    "lower_bounds": [0.0],
                    "upper_bounds": [1.0],
                    "indices": [0],
                },
                {
                    "type": "gaussian",
                    "mean": [0.0, 0.0],
                    "inv_covar": [[1.0, 0.0], [0.0, 1.0]],
                    "indices": [2, 3],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-5.0],
                    "upper_bounds": [5.0],
                    "indices": [1],
                },
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)
        prior = config.to_compound_prior()

        assert prior.n == 4
        assert len(prior.prior_components) == 3

    def test_empty_components_list(self) -> None:
        """Test configuration with empty components list."""
        config = CompoundPriorConfig(components=[])
        prior = config.to_compound_prior()

        assert isinstance(prior, CompoundPrior)
        assert prior.n == 0
        assert len(prior.prior_components) == 0

    def test_from_dict_preserves_order(self) -> None:
        """Test that from_dict preserves the order of components."""
        config_dict = {
            "components": [
                {
                    "type": "uniform",
                    "lower_bounds": [0.0],
                    "upper_bounds": [1.0],
                    "indices": [0],
                },
                {
                    "type": "gaussian",
                    "mean": [0.0],
                    "inv_covar": [[1.0]],
                    "indices": [1],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [2.0],
                    "upper_bounds": [3.0],
                    "indices": [2],
                },
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)

        assert isinstance(config.components[0], UniformPriorComponentConfig)
        assert isinstance(config.components[1], GaussianPriorComponentConfig)
        assert isinstance(config.components[2], UniformPriorComponentConfig)

        # Verify indices match
        assert config.components[0].indices == [0]
        assert config.components[1].indices == [1]
        assert config.components[2].indices == [2]

    def test_integration_yaml_like_structure(self) -> None:
        """Test with a realistic YAML-like configuration structure."""
        # Simulates what would be loaded from YAML
        config_dict = {
            "components": [
                {
                    "type": "gaussian",
                    "mean": [5800.0, 3200.0, 2700.0],
                    "inv_covar": [
                        [1e-6, 0.0, 0.0],
                        [0.0, 1e-6, 0.0],
                        [0.0, 0.0, 1e-6],
                    ],
                    "indices": [0, 1, 2],
                },
                {
                    "type": "uniform",
                    "lower_bounds": [-0.5, -0.5],
                    "upper_bounds": [0.5, 0.5],
                    "indices": [3, 4],
                },
            ]
        }

        config = CompoundPriorConfig.from_dict(config_dict)
        prior = config.to_compound_prior()

        assert prior.n == 5

        # Test with realistic values
        params = np.array([5800.0, 3200.0, 2700.0, 0.0, 0.0])
        log_prior = prior(params)
        assert log_prior == 0.0

        # Test with parameters outside uniform bounds
        params_out = np.array([5800.0, 3200.0, 2700.0, 1.0, 0.0])
        log_prior_out = prior(params_out)
        assert log_prior_out == -np.inf


class TestCompoundPriorConfigValidation:
    """Tests for CompoundPriorConfig __post_init__ validation."""

    def test_validation_missing_type_attribute(self) -> None:
        """Test validation catches component without type attribute."""

        # Create a component config without type attribute
        class InvalidConfig:
            indices = [0, 1]

        with pytest.raises(
            AttributeError, match="Component configuration missing a 'type'"
        ):
            CompoundPriorConfig(components=[InvalidConfig()])  # type: ignore

    def test_validation_invalid_prior_type_string(self) -> None:
        """Test validation catches invalid prior type string."""

        # Create a component with invalid type
        class InvalidConfig:
            type = "exponential"
            indices = [0, 1]

        with pytest.raises(ValueError, match="Unknown prior type"):
            CompoundPriorConfig(components=[InvalidConfig()])  # type: ignore

    def test_validation_overlapping_indices(self) -> None:
        """Test validation catches overlapping indices between components."""
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 1],
        )
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[-1.0, -2.0],
            upper_bounds=[1.0, 2.0],
            indices=[1, 2],  # Index 1 overlaps with gaussian_config
        )

        with pytest.raises(
            ValueError,
            match="Prior components must cover all parameter indices without overlap",
        ):
            CompoundPriorConfig(components=[gaussian_config, uniform_config])

    def test_validation_duplicate_indices_within_component(self) -> None:
        """Test validation catches duplicate indices within same component."""
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 0],  # Duplicate index
        )

        with pytest.raises(
            ValueError,
            match="Prior components must cover all parameter indices without overlap",
        ):
            CompoundPriorConfig(components=[gaussian_config])

    def test_validation_gap_in_indices(self) -> None:
        """Test validation catches gaps in parameter indices."""
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0],
            inv_covar=[[1.0, 0.0], [0.0, 1.0]],
            indices=[0, 1],
        )
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[-1.0],
            upper_bounds=[1.0],
            indices=[3],  # Gap at index 2
        )

        with pytest.raises(
            ValueError,
            match="Prior components must cover all parameter indices without overlap",
        ):
            CompoundPriorConfig(components=[gaussian_config, uniform_config])

    def test_validation_indices_not_starting_at_zero(self) -> None:
        """Test validation catches indices not starting from 0."""
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[-1.0, -2.0],
            upper_bounds=[1.0, 2.0],
            indices=[1, 2],  # Should start at 0
        )

        with pytest.raises(
            ValueError,
            match="Prior components must cover all parameter indices without overlap",
        ):
            CompoundPriorConfig(components=[uniform_config])

    def test_validation_unsorted_indices(self) -> None:
        """Test validation handles unsorted indices correctly."""
        # Validation should work regardless of order in each component
        uniform_config1 = UniformPriorComponentConfig(
            lower_bounds=[-1.0, -2.0],
            upper_bounds=[1.0, 2.0],
            indices=[1, 0],  # Unsorted but valid
        )
        uniform_config2 = UniformPriorComponentConfig(
            lower_bounds=[-3.0],
            upper_bounds=[3.0],
            indices=[2],
        )

        # This should not raise as indices cover [0, 1, 2] without gaps/overlap
        config = CompoundPriorConfig(components=[uniform_config1, uniform_config2])
        assert len(config.components) == 2

    def test_validation_case_insensitive_type(self) -> None:
        """Test that type validation is case-insensitive."""

        # Create a config with uppercase type
        class UpperCaseConfig:
            type = "GAUSSIAN"
            indices = [0]

        # Should work because validation uses .lower()
        config = CompoundPriorConfig(components=[UpperCaseConfig()])  # type: ignore
        assert len(config.components) == 1

    def test_validation_mixed_case_type(self) -> None:
        """Test that type validation handles mixed case."""

        class MixedCaseConfig:
            type = "GaUsSiAn"
            indices = [0]

        # Should work because validation uses .lower()
        config = CompoundPriorConfig(components=[MixedCaseConfig()])  # type: ignore
        assert len(config.components) == 1

    def test_validation_valid_single_component(self) -> None:
        """Test validation passes for valid single component."""
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0, 0.0, 0.0],
            inv_covar=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            indices=[0, 1, 2],
        )

        config = CompoundPriorConfig(components=[gaussian_config])
        assert len(config.components) == 1

    def test_validation_valid_multiple_components(self) -> None:
        """Test validation passes for valid multiple components."""
        gaussian_config = GaussianPriorComponentConfig(
            mean=[0.0],
            inv_covar=[[1.0]],
            indices=[0],
        )
        uniform_config = UniformPriorComponentConfig(
            lower_bounds=[-1.0, -2.0],
            upper_bounds=[1.0, 2.0],
            indices=[1, 2],
        )
        another_uniform = UniformPriorComponentConfig(
            lower_bounds=[-5.0],
            upper_bounds=[5.0],
            indices=[3],
        )

        config = CompoundPriorConfig(
            components=[gaussian_config, uniform_config, another_uniform]
        )
        assert len(config.components) == 3

    def test_validation_empty_components(self) -> None:
        """Test validation with empty components list."""
        config = CompoundPriorConfig(components=[])
        # Empty should be allowed (though not very useful)
        assert len(config.components) == 0
