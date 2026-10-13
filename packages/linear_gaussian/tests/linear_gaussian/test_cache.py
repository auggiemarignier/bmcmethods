"""Tests for linear_gaussian.cache."""

import numpy as np
from linear_gaussian import GaussianComponent
from linear_gaussian.cache import _CACHE, _ID_TOKEN_MAP, _prepare_problem, clear_cache


class TestPreparedProblem:
    def make_scalar_component(self, a=1.0, mu=0.0, c=1.0):
        A = np.array([[a]], dtype=float)
        mu = np.array([mu], dtype=float)
        C = np.array([[c]], dtype=float)
        return GaussianComponent(A, mu, C)

    def test_prepare_problem_both_sets(self):
        comp_inf = self.make_scalar_component(a=2.0, mu=0.0, c=2.0)
        comp_nui = self.make_scalar_component(a=3.0, mu=0.5, c=3.0)

        prepared = _prepare_problem([comp_inf], [comp_nui])

        assert prepared.M == 1
        assert prepared.A_I.shape == (1, 1)
        assert prepared.C_I.shape == (1, 1)
        assert prepared.mu_I.shape == (1,)
        assert prepared.mu_eta.shape == (1,)
        assert prepared.C_eta.shape == (1, 1)

        # Both covariance choleskies and Lambda should be computed
        assert prepared.L_C_I is not None
        assert prepared.L_C_eta is not None
        assert prepared.Lambda is not None
        assert prepared.L_Lambda is not None

        # Tokens should be present for the live components
        assert len(_ID_TOKEN_MAP) >= 2

    def test_prepare_problem_only_inferred(self):
        comp_inf = self.make_scalar_component(a=2.0, mu=0.0, c=2.0)
        prepared = _prepare_problem([comp_inf], [])

        assert prepared.M == 1
        assert prepared.L_C_I is not None
        # No nuisance => no C_eta cholesky and no Lambda
        assert prepared.L_C_eta is None
        assert prepared.Lambda is None
        assert prepared.L_Lambda is None

    def test_prepare_problem_only_nuisance(self):
        comp_nui = self.make_scalar_component(a=3.0, mu=0.5, c=3.0)
        prepared = _prepare_problem([], [comp_nui])

        assert prepared.M == 1
        # No inferred => no C_I cholesky and no Lambda
        assert prepared.L_C_I is None
        assert prepared.L_C_eta is not None
        assert prepared.Lambda is None
        assert prepared.L_Lambda is None

    def test_prepare_problem_cache_reuse_and_instance_isolation(self):
        comp_inf = self.make_scalar_component(a=2.0, mu=0.0, c=2.0)
        comp_nui = self.make_scalar_component(a=3.0, mu=0.5, c=3.0)

        p1 = _prepare_problem([comp_inf], [comp_nui])
        p2 = _prepare_problem([comp_inf], [comp_nui])
        assert p1 is p2

        # Different live instance, same numeric contents, must not reuse the same cache entry
        comp_inf2 = self.make_scalar_component(a=2.0, mu=0.0, c=2.0)
        p3 = _prepare_problem([comp_inf2], [comp_nui])
        assert p3 is not p1

    def test_clear_cache_clears_internal_structures(self):
        comp_inf = self.make_scalar_component(a=2.0, mu=0.0, c=2.0)
        comp_nui = self.make_scalar_component(a=3.0, mu=0.5, c=3.0)

        # populate caches
        _prepare_problem([comp_inf], [comp_nui])
        assert len(_CACHE) > 0
        assert len(_ID_TOKEN_MAP) > 0

        # clear and ensure internal structures are emptied
        clear_cache()
        assert len(_CACHE) == 0
        assert len(_ID_TOKEN_MAP) == 0
