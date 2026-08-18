"""Bayesian inference for linear combinations of Gaussian random variables.

Implements posterior mean, posterior covariance, and log-evidence for the
linear-Gaussian model

    d = A_I X_I + eta

where X_I is the inferred variable, A_I is the forward operator, and eta is
the aggregate nuisance contribution.  Both X_I and eta are Gaussian.
"""

import contextlib
import uuid
from dataclasses import InitVar, dataclass, field
from weakref import ref

import numpy as np
from numpy.typing import NDArray

type NDArrayFloat = NDArray[np.float64]


@dataclass(frozen=True)
class GaussianComponent:
    """A single Gaussian component in the linear-Gaussian model.

    This is a frozen dataclass and the internal arrays are read-only for caching safety.

    Parameters
    ----------
    A : ndarray, shape (M, N_j)
        Linear mapping from the N_j-dimensional latent variable to the
        M-dimensional observation space.
    mu : ndarray, shape (N_j,)
        Prior mean of the latent variable.
    C : ndarray, shape (N_j, N_j)
        Prior covariance of the latent variable.
    """

    A: NDArrayFloat
    mu: NDArrayFloat
    C: NDArrayFloat

    def __post_init__(self) -> None:
        """Validate Gaussian component array dimensions and compatibility."""

        A = np.array(self.A, copy=True)
        mu = np.array(self.mu, copy=True)
        C = np.array(self.C, copy=True)

        if A.ndim != 2:
            raise ValueError("A must be a 2D array")
        if mu.ndim != 1:
            raise ValueError("mu must be a 1D array")
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C must be a 2D square matrix")
        if mu.shape[0] != C.shape[0]:
            raise ValueError("mu and C dimensions are incompatible")
        if A.shape[1] != mu.shape[0]:
            raise ValueError("A columns and mu are incompatible")

        A.setflags(write=False)
        mu.setflags(write=False)
        C.setflags(write=False)
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "C", C)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _block_diag(matrices: list[NDArrayFloat]) -> NDArrayFloat:
    """Build a block-diagonal matrix from a list of square matrices."""
    sizes = [m.shape[0] for m in matrices]
    n = sum(sizes)
    result = np.zeros((n, n))
    offset = 0
    for m in matrices:
        s = m.shape[0]
        result[offset : offset + s, offset : offset + s] = m
        offset += s
    return result


def _calc_mu_eta(nuisance: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Compute the nuisance mean mu_eta = sum_m A_m mu_m."""
    if not nuisance:
        return np.zeros(M)
    return np.sum([c.A @ c.mu for c in nuisance], axis=0)


def _calc_C_eta(nuisance: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Compute the nuisance covariance C_eta = sum_m A_m C_m A_m^T."""
    if not nuisance:
        return np.zeros((M, M))
    return np.sum([c.A @ c.C @ c.A.T for c in nuisance], axis=0)


def _build_A_I(inferred: list[GaussianComponent], M: int) -> NDArrayFloat:
    """Build A_I = (A_1 | ... | A_k) by horizontal concatenation."""
    if not inferred:
        return np.zeros((M, 0))
    return np.hstack([c.A for c in inferred])


def _build_mu_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build mu_I = (mu_1; ...; mu_k) by vertical stacking."""
    if not inferred:
        return np.zeros(0)
    return np.concatenate([c.mu for c in inferred])


def _build_C_I(inferred: list[GaussianComponent]) -> NDArrayFloat:
    """Build C_I = diag(C_1, ..., C_k) as a block-diagonal matrix."""
    if not inferred:
        return np.zeros((0, 0))
    return _block_diag([c.C for c in inferred])


def _solve_cholesky(L: NDArrayFloat, rhs: NDArrayFloat) -> NDArrayFloat:
    y = np.linalg.solve(L, rhs)
    return np.linalg.solve(L.T, y)


def _solve_symmetric_system(
    L: NDArrayFloat | None, matrix: NDArrayFloat, rhs: NDArrayFloat
) -> NDArrayFloat:
    if L is not None:
        return _solve_cholesky(L, rhs)

    try:
        L_chol = np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Matrix is singular or not numerically positive definite."
            ) from exc
    else:
        return _solve_cholesky(L_chol, rhs)


def _calc_Lambda(
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    C_eta: NDArrayFloat,
    L_C_I: NDArrayFloat | None = None,
    L_C_eta: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Compute the posterior precision Lambda = C_I^{-1} + A_I^T C_eta^{-1} A_I."""
    identity = np.eye(C_I.shape[0], dtype=C_I.dtype)
    C_I_inv = _solve_symmetric_system(L_C_I, C_I, identity)
    C_eta_inv_A_I = _solve_symmetric_system(L_C_eta, C_eta, A_I)
    return C_I_inv + A_I.T @ C_eta_inv_A_I


def _calc_h(
    d: NDArrayFloat,
    mu_I: NDArrayFloat,
    C_I: NDArrayFloat,
    A_I: NDArrayFloat,
    mu_eta: NDArrayFloat,
    C_eta: NDArrayFloat,
    L_C_I: NDArrayFloat | None = None,
    L_C_eta: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Compute the information vector h = C_I^{-1} mu_I + A_I^T C_eta^{-1} (d - mu_eta).

    If Cholesky factors are provided they will be used for the solves.
    """
    C_I_inv_mu_I = _solve_symmetric_system(L_C_I, C_I, mu_I)
    C_eta_inv_residual = _solve_symmetric_system(L_C_eta, C_eta, d - mu_eta)
    return C_I_inv_mu_I + A_I.T @ C_eta_inv_residual


def _infer_M(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> int:
    """Infer the observation-space dimension M from the components."""
    all_components = inferred + nuisance
    return all_components[0].A.shape[0]


def _validate_inputs(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> None:
    """Validate that d, inferred and nuisance are mutually consistent."""
    if not inferred and not nuisance:
        raise ValueError("At least one of inferred or nuisance must be non-empty.")
    M = _infer_M(inferred, nuisance)
    for comp in inferred + nuisance:
        if comp.A.shape[0] != M:
            raise ValueError(
                f"All components must map to the same observation dimension M={M}, "
                f"but found A with shape {comp.A.shape}."
            )
    if d.ndim != 1:
        raise ValueError(f"d must be a 1D array, but has shape {d.shape}.")
    if d.shape[0] != M:
        raise ValueError(
            f"d has dimension {d.shape[0]} but components map to dimension M={M}."
        )


def _log_gaussian_density(
    x: NDArrayFloat,
    mean: NDArrayFloat,
    cov: NDArrayFloat,
) -> float:
    """Evaluate the log of N(x; mean, cov)."""
    M = x.shape[0]
    diff = x - mean

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        jitter_scale = float(np.trace(cov) / M)
        if not np.isfinite(jitter_scale) or jitter_scale <= 0:
            jitter_scale = 1.0
        jitter = max(1e-12 * jitter_scale, 1e-12)

        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(M))
        except np.linalg.LinAlgError:
            sign, log_det = np.linalg.slogdet(cov)
            if sign <= 0 or not np.isfinite(log_det):
                raise ValueError(
                    "Covariance matrix is not positive definite."
                ) from None
            quad = float(diff @ np.linalg.solve(cov, diff))
            return float(-0.5 * M * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad)

    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    y = np.linalg.solve(L, diff)
    quad = float(y @ y)
    return float(-0.5 * M * np.log(2 * np.pi) - 0.5 * log_det - 0.5 * quad)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calc_posterior_cov(
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Compute the posterior covariance of the inferred variables.

    Parameters
    ----------
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    C_post : ndarray, shape (N_I, N_I)
        Posterior covariance, where N_I = sum_j N_j over inferred components.

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    if not inferred:
        raise ValueError(
            "inferred must be non-empty to compute a posterior covariance."
        )
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior covariance; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )

    prepared = _prepare_problem(inferred, nuisance)
    if prepared.Lambda is None:
        raise RuntimeError("_prepare_problem did not compute Lambda.")
    identity = np.eye(prepared.Lambda.shape[0], dtype=prepared.Lambda.dtype)
    return _solve_symmetric_system(prepared.L_Lambda, prepared.Lambda, identity)


def calc_posterior_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Compute the posterior mean of the inferred variables.

    Parameters
    ----------
    d : ndarray, shape (M,)
        Observed data vector.
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    mu_post : ndarray, shape (N_I,)
        Posterior mean, where N_I = sum_j N_j over inferred components.

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    if not inferred:
        raise ValueError("inferred must be non-empty to compute a posterior mean.")
    if not nuisance:
        raise ValueError(
            "nuisance must be non-empty to compute the posterior mean; "
            "the noise-free case (nuisance=[]) leads to a degenerate posterior."
        )
    _validate_inputs(d, inferred, nuisance)

    prepared = _prepare_problem(inferred, nuisance)
    if prepared.Lambda is None:
        raise RuntimeError("_prepare_problem did not compute Lambda.")
    h = _calc_h(
        d,
        prepared.mu_I,
        prepared.C_I,
        prepared.A_I,
        prepared.mu_eta,
        prepared.C_eta,
        prepared.L_C_I,
        prepared.L_C_eta,
    )
    return _solve_symmetric_system(prepared.L_Lambda, prepared.Lambda, h)


def calc_log_evidence(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> float:
    """Compute the log marginal likelihood (log evidence) log p(d).

    The evidence is the marginal likelihood obtained by integrating out all
    latent variables.  Because the sum of linear transformations of independent
    Gaussians is itself Gaussian, the marginal distribution of the data is

        d ~ N(A_I mu_I + mu_eta,  A_I C_I A_I^T + C_eta)

    where nuisance variables have been analytically marginalised to yield
    mu_eta and C_eta.

    Parameters
    ----------
    d : ndarray, shape (M,)
        Observed data vector.
    inferred : list of GaussianComponent
        The inferred components.  May be empty, in which case the evidence
        reduces to the nuisance marginal N(d; mu_eta, C_eta).
    nuisance : list of GaussianComponent
        The nuisance components.  May be empty, in which case the evidence
        uses only the inferred prior marginal N(d; A_I mu_I, A_I C_I A_I^T).

    Returns
    -------
    log_Z : float
        Log marginal likelihood log p(d).

    Raises
    ------
    ValueError
        If both ``inferred`` and ``nuisance`` are empty.
    """
    _validate_inputs(d, inferred, nuisance)
    prepared = _prepare_problem(inferred, nuisance)

    if not inferred:
        # Only nuisance: d ~ N(mu_eta, C_eta)
        mu_marginal = prepared.mu_eta
        C_marginal = prepared.C_eta
    elif not nuisance:
        # Only inferred, no noise: d ~ N(A_I mu_I, A_I C_I A_I^T)
        A_I = prepared.A_I
        mu_I = prepared.mu_I
        C_I = prepared.C_I
        mu_marginal = A_I @ mu_I
        C_marginal = A_I @ C_I @ A_I.T
    else:
        # General case
        A_I = prepared.A_I
        mu_I = prepared.mu_I
        C_I = prepared.C_I
        mu_eta = prepared.mu_eta
        C_eta = prepared.C_eta
        mu_marginal = A_I @ mu_I + mu_eta
        C_marginal = A_I @ C_I @ A_I.T + C_eta

    return _log_gaussian_density(d, mu_marginal, C_marginal)


def calc_posterior_predictive_mean(
    d: NDArrayFloat,
    inferred: list[GaussianComponent],
    nuisance: list[GaussianComponent],
) -> NDArrayFloat:
    """Compute the mean of the posterior predictive distribution.

    Computes the mean of the posterior predictive distribution d^* | d where d^* is a new replicate of the original data.

        mu_pred = A_I mu_post + mu_eta

    Parameters
    ----------
    d : ndarray, shape (M,)
        Observed data vector.
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    mu_pred : ndarray, shape (M,)
        Posterior predictive mean

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    mu_post = calc_posterior_mean(d, inferred, nuisance)
    prepared = _prepare_problem(inferred, nuisance)
    return prepared.A_I @ mu_post + prepared.mu_eta


def calc_posterior_predictive_cov(
    inferred: list[GaussianComponent], nuisance: list[GaussianComponent]
) -> NDArrayFloat:
    """Compute the covariance of the posterior predictive distribution.

    Computes the covariance of the posterior predictive distribution d^* | d where d^* is a new replicate of the original data.

        C_pred = A_I C_post A_I^T + C_eta

    Parameters
    ----------
    inferred : list of GaussianComponent
        The components whose latent variables are to be inferred.
    nuisance : list of GaussianComponent
        The components that are marginalised out (provide the noise model).

    Returns
    -------
    C_pred : ndarray, shape (M, M)
        Posterior predictive covariance of a new replicate d^*, where M is the
        original data length.

    Raises
    ------
    ValueError
        If ``inferred`` is empty (no variables to infer) or ``nuisance`` is
        empty (degenerate noise-free model).
    """
    C_post = calc_posterior_cov(inferred, nuisance)
    prepared = _prepare_problem(inferred, nuisance)
    return prepared.A_I @ C_post @ prepared.A_I.T + prepared.C_eta


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


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
