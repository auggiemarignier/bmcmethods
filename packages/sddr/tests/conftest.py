"""Common test fixtures and configurations."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator fixture."""
    return np.random.default_rng(42)
