"""Cache for strategy-independent arrays assembled from Gaussian components."""

import uuid
from weakref import ref

import numpy as np

from .models import GaussianComponent

_Prepared = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_PrecisionCache = tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
]
_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], _Prepared] = {}
_PRECISION_CACHE: dict[tuple[tuple[str, ...], tuple[str, ...]], _PrecisionCache] = {}
_ID_TOKEN_MAP: dict[int, tuple[ref, str]] = {}
_MAX_CACHE_SIZE = 128


def _token_for(comp: GaussianComponent) -> str:
    """Return a stable token for this component instance while it is alive."""
    key = id(comp)
    entry = _ID_TOKEN_MAP.get(key)
    if entry is not None:
        component_ref, token = entry
        if component_ref() is comp:
            return token

    token = uuid.uuid4().hex
    # remove mapping when object is GC'd
    _ID_TOKEN_MAP[key] = (
        ref(comp, lambda _ref, k=key: _ID_TOKEN_MAP.pop(k, None)),
        token,
    )
    return token


def _prep_key(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(_token_for(c) for c in inferred),
        tuple(_token_for(c) for c in nuisance),
    )


def get_prepared(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> _Prepared | None:
    """Look up shared arrays for these component instances."""
    return _CACHE.get(_prep_key(inferred, nuisance))


def store_prepared(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    prepared: _Prepared,
) -> None:
    """Store shared arrays, evicting the oldest entry at capacity."""
    if len(_CACHE) >= _MAX_CACHE_SIZE:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[_prep_key(inferred, nuisance)] = prepared


def get_precision(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> _PrecisionCache | None:
    """Look up cached precision and its Cholesky factor."""
    return _PRECISION_CACHE.get(_prep_key(inferred, nuisance))


def store_precision(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
    precision: _PrecisionCache,
) -> None:
    """Store precision matrices and factors, evicting at capacity."""
    if len(_PRECISION_CACHE) >= _MAX_CACHE_SIZE:
        _PRECISION_CACHE.pop(next(iter(_PRECISION_CACHE)))
    _PRECISION_CACHE[_prep_key(inferred, nuisance)] = precision


def clear_cache() -> None:
    """Clear shared preparation and precision strategy caches."""
    _CACHE.clear()
    _PRECISION_CACHE.clear()
    _ID_TOKEN_MAP.clear()
