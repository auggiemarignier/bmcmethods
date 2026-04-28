from multiprocessing import Pool
from typing import Self

import numpy as np
import pytest
from sampling import _worker
from sampling._worker import DummyPool, init_worker, logl, logp
from sampling.likelihood._base import (
    ForwardBase,
    ForwardGradientBase,
    IdentityForward,
    IdentityState,
    LikelihoodBase,
    NoForwardGradient,
)


class DummyLikelihood(LikelihoodBase[int]):
    """Simple picklable likelihood with an integer state multiplier."""

    def __init__(self, factor: int) -> None:
        self.state = factor

    @classmethod
    def from_state(
        cls,
        state: int,
        *,
        forward: ForwardBase | None = None,
        forward_gradient: ForwardGradientBase | None = None,
    ) -> Self:
        return cls(state)

    def __call__(self, model_params: np.ndarray) -> float:
        return float(np.sum(model_params) * self.state)


@pytest.fixture
def likelihood() -> DummyLikelihood:
    return DummyLikelihood(3)


class DummyPrior:
    """Simple prior object implementing the PriorFunction protocol."""

    def __init__(self, offset: float = 0.0, n: int = 1) -> None:
        self.config_params: list[np.ndarray] = []
        self.n = n
        self.offset = offset

    def __call__(self, model_params: np.ndarray) -> np.ndarray:
        return np.asarray(np.sum(model_params) + self.offset)

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        return np.zeros_like(model_params)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(num_samples, self.n))


@pytest.fixture
def prior() -> DummyPrior:
    return DummyPrior(0.5)


def test_worker_pool_logl_logp(likelihood: DummyLikelihood, prior: DummyPrior) -> None:
    """Demonstrate initialising workers with `init_worker` and using `logl`/`logp`.

    This spawns a small Pool with the module initializer and maps `logl`/`logp`
    across simple inputs to show that the globals are reconstructed in each
    worker process.
    """

    # Initialise worker processes with DummyLikelihood state=3 and DummyPrior offset=0.5
    with Pool(
        processes=2,
        initializer=init_worker,
        initargs=(
            likelihood.__class__,
            likelihood.state,
            prior,
            IdentityForward,
            IdentityState(),
            NoForwardGradient,
            IdentityState(),
        ),
    ) as p:
        inputs = [np.array([1.0]), np.array([2.0])]
        logl_results = p.map(logl, inputs)
        logp_results = p.map(logp, inputs)

    assert logl_results == [3.0, 6.0]
    assert logp_results == [1.5, 2.5]


def test_logl_raises_when_uninitialised() -> None:
    # ensure globals are unset in the main process
    _worker.WORKER_LIKELIHOOD = None
    with pytest.raises(RuntimeError, match="likelihood"):
        logl(np.array([1.0]))


def test_logp_raises_when_uninitialised() -> None:
    # ensure globals are unset in the main process
    _worker.WORKER_PRIOR = None
    with pytest.raises(RuntimeError, match="prior"):
        logp(np.array([1.0]))


def test_make_pool_serial_calls_init_worker(
    monkeypatch, likelihood: DummyLikelihood, prior: DummyPrior
):
    called = {}

    def fake_init_worker(*args):
        called["args"] = args

    monkeypatch.setattr(_worker, "init_worker", fake_init_worker)

    pool = _worker.make_pool(False, likelihood, prior)

    assert isinstance(pool, DummyPool)
    assert called["args"][0] is DummyLikelihood


class FakePool:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_make_pool_parallel_constructs_pool(
    monkeypatch, likelihood: DummyLikelihood, prior: DummyPrior
):
    captured = {}

    def fake_pool(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakePool(*args, **kwargs)

    monkeypatch.setattr(_worker, "Pool", fake_pool)
    monkeypatch.setattr(_worker, "init_worker", lambda *a, **k: None)

    pool = _worker.make_pool(4, likelihood, prior)

    assert isinstance(pool, FakePool)
    assert captured["kwargs"]["processes"] == 4
    assert captured["kwargs"]["initializer"] is _worker.init_worker
    assert captured["kwargs"]["initargs"][0] is DummyLikelihood
    assert captured["kwargs"]["initargs"][1] == 3


def test_make_pool_true_uses_cpu_count(
    monkeypatch, likelihood: DummyLikelihood, prior: DummyPrior
):
    monkeypatch.setattr(_worker, "Pool", lambda **kwargs: kwargs)
    monkeypatch.setattr(_worker, "init_worker", lambda *a, **k: None)
    monkeypatch.setattr(_worker.os, "cpu_count", lambda: 8)

    pool = _worker.make_pool(True, likelihood, prior)

    assert pool["processes"] == 8


class TestDummyPool:
    """Tests for DummyPool ensuring behavioural parity with multiprocessing.Pool subset.

    All tests compare DummyPool against multiprocessing.Pool for the supported
    methods: map, starmap, imap plus context manager behaviour. The goal is to
    verify order, return types and exception propagation so calling code can
    switch between serial and parallel execution without branching.
    """

    from multiprocessing import Pool

    def _square(self, x: int) -> int:
        return x * x

    def _add(self, x: int, y: int) -> int:
        return x + y

    def test_map_equivalence(self) -> None:
        """Map produces identical ordered list of results as ``multiprocessing.Pool.map``."""
        data = list(range(10))

        with Pool(processes=2) as p:
            parallel = p.map(self._square, data)

        serial = DummyPool().map(self._square, data)

        assert parallel == serial
        assert isinstance(serial, list)
        assert all(isinstance(v, int) for v in serial)

    def test_map_empty_iterable(self) -> None:
        """Map on an empty iterable returns an empty list (parity with Pool)."""
        with Pool(processes=2) as p:
            parallel = p.map(self._square, [])
        serial = DummyPool().map(self._square, [])
        assert parallel == serial == []

    def test_starmap_equivalence(self) -> None:
        """Starmap unpacks tuples identically to ``multiprocessing.Pool.starmap``."""
        args = [(i, i + 1) for i in range(5)]
        with Pool(processes=2) as p:
            parallel = p.starmap(self._add, args)
        serial = DummyPool().starmap(self._add, args)
        assert parallel == serial
        assert isinstance(serial, list)

    def test_starmap_empty_iterable(self) -> None:
        """Starmap over an empty iterable returns an empty list (parity)."""
        with Pool(processes=2) as p:
            parallel = p.starmap(self._add, [])
        serial = DummyPool().starmap(self._add, [])
        assert parallel == serial == []

    def test_imap_equivalence(self) -> None:
        """Imap yields identical sequence of results as ``multiprocessing.Pool.imap``."""
        data = list(range(7))
        with Pool(processes=2) as p:
            parallel_iter = p.imap(self._square, data)
            parallel = list(parallel_iter)
        serial_iter = DummyPool().imap(self._square, data)
        assert hasattr(serial_iter, "__iter__")
        serial = list(serial_iter)
        assert parallel == serial

    def test_context_manager_returns_self(self) -> None:
        """Context manager returns the instance and permits method calls inside block."""
        with DummyPool() as pool:
            assert isinstance(pool, DummyPool)
            result = pool.map(self._square, [3])
            assert result == [9]

    def test_close_and_join_noop(self) -> None:
        """close() and join() exist and perform no operation without raising."""
        pool = DummyPool()
        pool.close()
        pool.join()

    def test_exception_propagation_in_context(self) -> None:
        """Exceptions raised inside the ``with`` block propagate (not suppressed)."""

        def boom(x: int) -> int:  # pragma: no cover - behaviour assertion
            if x == 2:
                raise ValueError("bang")
            return x

        with pytest.raises(ValueError), DummyPool() as pool:
            pool.map(boom, [0, 1, 2, 3])

    def test_dummy_pool_does_not_spawn_processes(self) -> None:
        """Map performs side effects locally (no separate worker processes)."""
        counter = {"n": 0}

        def bump(x: int) -> int:
            counter["n"] += 1
            return x

        data = list(range(5))
        result = DummyPool().map(bump, data)
        assert result == data
        assert counter["n"] == len(data)

    def test_imap_streaming(self) -> None:
        """Imap yields a lazy iterator that can be partially consumed and resumed."""
        items = list(range(5))
        gen = DummyPool().imap(self._square, items)
        first_two = [next(gen), next(gen)]
        assert first_two == [0, 1]
        rest = list(gen)
        assert rest == [4, 9, 16]
