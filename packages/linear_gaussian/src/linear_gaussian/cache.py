"""Prepared-problem construction and cache lifecycle."""

import contextlib
import uuid
from dataclasses import InitVar, dataclass, field
from weakref import ref

import numpy as np

from .models import GaussianComponent
from .precision import (
    _build_A_I,
    _build_C_I,
    _build_mu_I,
    _calc_C_eta,
    _calc_Lambda,
    _calc_mu_eta,
)


def _infer_M(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> int:
    """Infer the observation-space dimension M from the components."""
    return (inferred + nuisance)[0].A.shape[0]


@dataclass(frozen=True)
class _PreparedProblem:
    M: int
    A_I: np.ndarray
    C_I: np.ndarray
    mu_I: np.ndarray
    mu_eta: np.ndarray
    C_eta: np.ndarray
    Lambda: np.ndarray | None = field(default=None, init=False)
    L_C_I: np.ndarray | None = field(default=None, init=False)
    L_C_eta: np.ndarray | None = field(default=None, init=False)
    L_Lambda: np.ndarray | None = field(default=None, init=False)
    _no_nuisance: InitVar[bool] = False
    _no_inferred: InitVar[bool] = False

    def __post_init__(self, _no_nuisance: bool, _no_inferred: bool) -> None:
        if not _no_inferred:
            with contextlib.suppress(np.linalg.LinAlgError):
                object.__setattr__(self, "L_C_I", np.linalg.cholesky(self.C_I))
        if not _no_nuisance:
            with contextlib.suppress(np.linalg.LinAlgError):
                object.__setattr__(self, "L_C_eta", np.linalg.cholesky(self.C_eta))

        if not _no_inferred and not _no_nuisance:
            object.__setattr__(
                self,
                "Lambda",
                _calc_Lambda(self.C_I, self.A_I, self.C_eta, self.L_C_I, self.L_C_eta),
            )

        if self.Lambda is not None:
            with contextlib.suppress(np.linalg.LinAlgError):
                object.__setattr__(self, "L_Lambda", np.linalg.cholesky(self.Lambda))


_CACHE: dict[tuple, _PreparedProblem] = {}
_ID_TOKEN_MAP: dict[int, tuple[ref, str]] = {}


def _token_for(comp: GaussianComponent) -> str:
    """
    Map from object id() → (weakref, token) used to derive stable, cheap per-instance keys for the cache without changing the GaussianComponent dataclass.

    Why this exists

        We need a stable, fast key for each live component instance so the cache can reuse prepared results without hashing large NumPy arrays.
        Using id() alone is unsafe because Python can reuse integer object ids after an object is garbage-collected; that could cause a new object to accidentally reuse an old cache entry.

    What is stored

        Key: id(comp) (an integer).
        Value: a tuple (ref, token) where:
            ref is a weakref.ref pointing to the original object, with a callback that removes the mapping when the object is collected.
            token is a short stable identifier (a UUID hex string) generated once for the object and used as the actual cache key component.

    How it works (simple)

        When we need a token for a component, we look up _ID_TOKEN_MAP[id(comp)].
        If the entry exists and ref() still returns the same object, we reuse token.
        If the entry is missing or ref() is None/not the same object, we create a new token and store (weakref.ref(comp, _cleanup_callback), token) under id(comp).
        The weakref callback removes the dictionary entry automatically when the object is garbage-collected, avoiding memory leaks and stale mappings.

    Important properties / caveats

        No heavy array/content hashing is performed — only id() checks and one-time UUID generation per live instance.
        The map does not keep strong references to components (the weakref avoids that).
        This design prevents id-reuse bugs because we validate the weakref referent before returning a stored token.
        Not thread-safe: if you access _ID_TOKEN_MAP concurrently from multiple threads, race conditions may occur. Add a simple threading.Lock around _token_for if multi-threaded access is expected.
        Tokens are stable only for the lifetime of the instance; they do not persist across process restarts or after the object is collected.

    Example (what _token_for(comp) guarantees)

        For the same live comp object: repeated _token_for(comp) calls return the same token.
        For a different object that happens to have the same id() (after GC/reuse): a new token will be generated and stored — the old token will not be reused.
    """
    key = id(comp)
    entry = _ID_TOKEN_MAP.get(key)
    if entry:
        r, token = entry
        obj = r()
        if obj is comp:
            return token
    token = uuid.uuid4().hex
    # remove mapping when object is GC'd
    _ID_TOKEN_MAP[key] = (
        ref(comp, lambda _r, k=key: _ID_TOKEN_MAP.pop(k, None)),
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


def _prepare_problem(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> _PreparedProblem:
    key = _prep_key(inferred, nuisance)
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    M = _infer_M(inferred, nuisance)
    A_I = _build_A_I(inferred, M)
    C_I = _build_C_I(inferred)
    mu_I = _build_mu_I(inferred)
    mu_eta = _calc_mu_eta(nuisance, M)
    C_eta = _calc_C_eta(nuisance, M)
    prepared = _PreparedProblem(
        M,
        A_I,
        C_I,
        mu_I,
        mu_eta,
        C_eta,
        _no_nuisance=len(nuisance) == 0,
        _no_inferred=len(inferred) == 0,
    )

    if len(_CACHE) >= 128:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = prepared
    return prepared


def clear_cache() -> None:
    """Clear the prepared-problem cache and associated ID->token map.

    Not thread-safe: if you call this concurrently with other cache access from multiple
    threads, wrap cache operations in a lock.
    """
    _CACHE.clear()
    _ID_TOKEN_MAP.clear()
