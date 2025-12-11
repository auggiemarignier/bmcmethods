"""Common prior utilities."""

from collections.abc import Sequence

import numpy as np


def _normalise_indices(
    indices: Sequence[int] | slice | np.ndarray, n_total: int
) -> np.ndarray:
    """Convert various types of indices to a sorted numpy array."""
    if isinstance(indices, slice):
        start, stop, step = _unpack_slice(indices, n_total)
        return np.sort(np.arange(start, stop, step))
    else:
        return np.sort(np.asarray(indices, dtype=int))


def _unpack_slice(s: slice, n_total: int) -> tuple[int, int, int]:
    """Unpack the slice indices into start, stop, step."""
    start = s.start or 0
    stop = s.stop or n_total
    step = s.step or 1
    return start, stop, step
