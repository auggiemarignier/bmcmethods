"""Tests for DummyPool ensuring behavioural parity with multiprocessing.Pool subset.

All tests compare DummyPool against multiprocessing.Pool for the supported
methods: map, starmap, imap plus context manager behaviour. The goal is to
verify order, return types and exception propagation so calling code can
switch between serial and parallel execution without branching.
"""

from multiprocessing import Pool

import pytest
from sampling._util import DummyPool


def _square(x: int) -> int:
    return x * x


def _add(x: int, y: int) -> int:
    return x + y


def test_map_equivalence() -> None:
    """Map produces identical ordered list of results as ``multiprocessing.Pool.map``."""
    data = list(range(10))

    with Pool(processes=2) as p:
        parallel = p.map(_square, data)

    serial = DummyPool().map(_square, data)

    assert parallel == serial
    assert isinstance(serial, list)
    assert all(isinstance(v, int) for v in serial)


def test_map_empty_iterable() -> None:
    """Map on an empty iterable returns an empty list (parity with Pool)."""
    with Pool(processes=2) as p:
        parallel = p.map(_square, [])
    serial = DummyPool().map(_square, [])
    assert parallel == serial == []


def test_starmap_equivalence() -> None:
    """Starmap unpacks tuples identically to ``multiprocessing.Pool.starmap``."""
    args = [(i, i + 1) for i in range(5)]
    with Pool(processes=2) as p:
        parallel = p.starmap(_add, args)
    serial = DummyPool().starmap(_add, args)
    assert parallel == serial
    assert isinstance(serial, list)


def test_starmap_empty_iterable() -> None:
    """Starmap over an empty iterable returns an empty list (parity)."""
    with Pool(processes=2) as p:
        parallel = p.starmap(_add, [])
    serial = DummyPool().starmap(_add, [])
    assert parallel == serial == []


def test_imap_equivalence() -> None:
    """Imap yields identical sequence of results as ``multiprocessing.Pool.imap``."""
    data = list(range(7))
    with Pool(processes=2) as p:
        parallel_iter = p.imap(_square, data)
        parallel = list(parallel_iter)
    serial_iter = DummyPool().imap(_square, data)
    assert hasattr(serial_iter, "__iter__")
    serial = list(serial_iter)
    assert parallel == serial


def test_context_manager_returns_self() -> None:
    """Context manager returns the instance and permits method calls inside block."""
    with DummyPool() as pool:
        assert isinstance(pool, DummyPool)
        result = pool.map(_square, [3])
        assert result == [9]


def test_close_and_join_noop() -> None:
    """close() and join() exist and perform no operation without raising."""
    pool = DummyPool()
    pool.close()
    pool.join()


def test_exception_propagation_in_context() -> None:
    """Exceptions raised inside the ``with`` block propagate (not suppressed)."""

    def boom(x: int) -> int:  # pragma: no cover - behaviour assertion
        if x == 2:
            raise ValueError("bang")
        return x

    with pytest.raises(ValueError), DummyPool() as pool:
        pool.map(boom, [0, 1, 2, 3])


def test_dummy_pool_does_not_spawn_processes() -> None:
    """Map performs side effects locally (no separate worker processes)."""
    counter = {"n": 0}

    def bump(x: int) -> int:
        counter["n"] += 1
        return x

    data = list(range(5))
    result = DummyPool().map(bump, data)
    assert result == data
    assert counter["n"] == len(data)


def test_imap_streaming() -> None:
    """Imap yields a lazy iterator that can be partially consumed and resumed."""
    items = list(range(5))
    gen = DummyPool().imap(_square, items)
    first_two = [next(gen), next(gen)]
    assert first_two == [0, 1]
    rest = list(gen)
    assert rest == [4, 9, 16]
