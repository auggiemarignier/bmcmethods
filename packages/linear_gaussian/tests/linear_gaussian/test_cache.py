"""Tests for strategy-independent preparation caching."""

import numpy as np
from linear_gaussian import GaussianComponent, clear_cache
from linear_gaussian.cache import _CACHE, _ID_TOKEN_MAP, _PRECISION_CACHE
from linear_gaussian.common import prepare


class TestPreparedArrays:
    def setup_method(self):
        clear_cache()

    def teardown_method(self):
        clear_cache()

    @staticmethod
    def make_component(a=1.0, mu=0.0, c=1.0):
        return GaussianComponent(
            np.array([[a]], dtype=float),
            np.array([mu], dtype=float),
            np.array([[c]], dtype=float),
        )

    def test_prepare_assembles_arrays_for_both_component_sets(self):
        inferred = [self.make_component(a=2.0, mu=0.4, c=2.0)]
        nuisance = [self.make_component(a=3.0, mu=0.5, c=3.0)]

        arrays = prepare(inferred, nuisance)

        expected = (
            np.array([[2.0]]),
            np.array([0.4]),
            np.array([[2.0]]),
            np.array([1.5]),
            np.array([[27.0]]),
        )
        for actual, expected_array in zip(arrays, expected, strict=True):
            np.testing.assert_allclose(actual, expected_array)

    def test_reuses_preparation_and_separates_component_instances(self):
        inferred = [self.make_component(a=2.0, mu=0.0, c=2.0)]
        nuisance = [self.make_component(a=3.0, mu=0.5, c=3.0)]

        first = prepare(inferred, nuisance)
        second = prepare(inferred, nuisance)
        assert second is first

        same_values_new_instance = [self.make_component(a=2.0, mu=0.0, c=2.0)]
        third = prepare(same_values_new_instance, nuisance)
        assert third is not first

    def test_strategy_calls_share_the_same_prepared_arrays(self):
        from linear_gaussian.data_space import posterior_cov as data_cov
        from linear_gaussian.precision import posterior_cov as precision_cov

        inferred = [self.make_component(a=1.0, c=2.0)]
        nuisance = [self.make_component(a=1.0, c=1.0)]

        prepare(inferred, nuisance)
        assert len(_CACHE) == 1
        precision_cov(inferred, nuisance)
        data_cov(inferred, nuisance)
        assert len(_CACHE) == 1

    def test_clear_cache_clears_cached_arrays_and_tokens(self):
        inferred = [self.make_component()]
        nuisance = [self.make_component()]

        prepare(inferred, nuisance)
        assert _CACHE
        assert _ID_TOKEN_MAP

        clear_cache()
        assert not _CACHE
        assert not _PRECISION_CACHE
        assert not _ID_TOKEN_MAP

    def test_precision_factorization_is_reused_by_mean_and_covariance(
        self, monkeypatch
    ):
        from linear_gaussian import precision

        inferred = [self.make_component(a=1.0, c=2.0)]
        nuisance = [self.make_component(a=1.0, c=1.0)]
        original = precision._calc_Lambda
        original_cholesky = precision._cholesky_or_none
        calls = 0
        cholesky_calls = 0

        def count_calculations(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        def count_cholesky(matrix):
            nonlocal cholesky_calls
            cholesky_calls += 1
            return original_cholesky(matrix)

        monkeypatch.setattr(precision, "_calc_Lambda", count_calculations)
        monkeypatch.setattr(precision, "_cholesky_or_none", count_cholesky)
        precision.posterior_mean(np.array([0.5]), inferred, nuisance)
        first = next(iter(_PRECISION_CACHE.values()))
        precision.posterior_cov(inferred, nuisance)

        assert calls == 1
        assert cholesky_calls == 2
        assert len(_PRECISION_CACHE) == 1
        assert next(iter(_PRECISION_CACHE.values())) is first

        new_inferred = [self.make_component(a=1.0, c=2.0)]
        precision.posterior_cov(new_inferred, nuisance)
        assert len(_PRECISION_CACHE) == 2

    def test_cache_evicts_oldest_entry_at_capacity(self):
        clear_cache()
        inferred = [self.make_component()]
        nuisance = [self.make_component()]
        first = prepare(inferred, nuisance)

        from linear_gaussian import cache

        original_max_size = cache._MAX_CACHE_SIZE
        cache._MAX_CACHE_SIZE = 1
        try:
            changed_inferred = [self.make_component(a=2.0)]
            prepare(changed_inferred, nuisance)
            assert len(_CACHE) == 1
            assert all(value is not first for value in _CACHE.values())
        finally:
            cache._MAX_CACHE_SIZE = original_max_size
            clear_cache()
