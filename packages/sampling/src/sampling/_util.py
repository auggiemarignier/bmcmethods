"""Utilities for parallel and serial execution abstraction."""

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class DummyPool:
    """Serial stand-in for ``multiprocessing.Pool`` used when parallelisation is disabled.

    Provides a small, synchronous subset of the Pool API so calling code can
    switch between parallel and serial execution without branching. All work
    is executed immediately in the calling process; there is no task queue,
    no worker processes and therefore no need to ``close`` or ``join``.
    """

    def map(self, func: Callable[[T], U], iterable: Iterable[T]) -> list[U]:
        """Apply ``func`` to each item in ``iterable`` and return a list of results.

        Mirrors ``multiprocessing.Pool.map`` but executes serially. Order of
        results matches the order of the input iterable.
        """
        return [func(x) for x in iterable]

    def starmap(
        self, func: Callable[..., U], iterable: Iterable[Sequence[Any]]
    ) -> list[U]:
        """Apply ``func`` to argument sequences from ``iterable`` unpacking each.

        Serial analogue of ``multiprocessing.Pool.starmap``. Each element of
        ``iterable`` must be a sequence whose contents are passed as positional
        arguments to ``func``.
        """
        return [func(*args) for args in iterable]

    def imap(self, func: Callable[[T], U], iterable: Iterable[T]) -> Iterator[U]:
        """Yield results of applying ``func`` to each item in ``iterable`` lazily.

        Serial equivalent of ``multiprocessing.Pool.imap``. Useful when the
        consumer can stream results and does not need them all materialised.
        """
        for x in iterable:
            yield func(x)

    def __enter__(self) -> "DummyPool":
        """Return self to support ``with DummyPool() as pool:`` usage."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        """Context manager exit hook; does not suppress exceptions.

        Returning ``False`` propagates any exception raised inside the block.
        """
        return False

    def close(self) -> None:
        """Mirror of ``Pool.close()``. No operation in the dummy implementation."""
        pass

    def join(self) -> None:
        """Mirror of ``Pool.join()``. No operation—there are no worker processes."""
        pass
