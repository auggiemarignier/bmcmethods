"""Tests for linear_gaussian.models."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from linear_gaussian import GaussianComponent


class TestGaussianComponent:
    def test_valid_construction(self):
        gc = GaussianComponent(A=np.ones((3, 2)), mu=np.zeros(2), C=np.eye(2))
        assert gc.A.shape == (3, 2)
        assert gc.mu.shape == (2,)
        assert gc.C.shape == (2, 2)

    def test_A_must_be_2d(self):
        with pytest.raises(ValueError, match="A must be a 2D array"):
            GaussianComponent(A=np.ones(3), mu=np.zeros(3), C=np.eye(3))

    def test_mu_must_be_1d(self):
        with pytest.raises(ValueError, match="mu must be a 1D array"):
            GaussianComponent(A=np.eye(3), mu=np.ones((3, 1)), C=np.eye(3))

    def test_C_must_be_2d_square(self):
        with pytest.raises(ValueError, match="C must be a 2D square matrix"):
            GaussianComponent(A=np.eye(3), mu=np.zeros(3), C=np.ones((3, 4)))

    def test_mu_C_dimensions_must_match(self):
        with pytest.raises(ValueError, match="mu and C dimensions are incompatible"):
            GaussianComponent(A=np.ones((3, 2)), mu=np.zeros(3), C=np.eye(2))

    def test_A_columns_must_match_mu(self):
        with pytest.raises(ValueError, match="A columns and mu are incompatible"):
            GaussianComponent(A=np.ones((3, 4)), mu=np.zeros(2), C=np.eye(2))

    def _make_gc(self):
        return GaussianComponent(A=np.eye(2), mu=np.zeros(2), C=np.eye(2))

    def test_dataclass_is_frozen(self):
        gc = self._make_gc()
        with pytest.raises(FrozenInstanceError):
            gc.A = np.zeros((2, 2))

    def test_internal_arrays_are_readonly(self):
        gc = self._make_gc()
        assert gc.A.flags.writeable is False
        assert gc.mu.flags.writeable is False
        assert gc.C.flags.writeable is False

        with pytest.raises(ValueError):
            gc.A[0, 0] = 1.0
