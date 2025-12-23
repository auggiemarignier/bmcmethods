"""Estimate the evidence using the harmonic mean estimator from reciprocal importance sampling."""


# The implementations here are not the most computationally efficient.
# Particularly, there are lots of repeated calculations e.g. when computing the variances.
# However, they are structured for clarity and correctness first.
# Performance is not really a concern, as the computationally expensive part of RIS is first sampling the posterior and then fitting the model.
# The evidence estimation is relatively cheap in comparison.

from enum import Enum, auto
from warnings import warn

import numpy as np
import scipy.special as sp


class Shifting(Enum):
    """
    Enumeration to define which log-space shifting to adopt.

    Different choices may prove optimal for certain settings.
    """

    MEAN_SHIFT = auto()
    MAX_SHIFT = auto()
    MIN_SHIFT = auto()
    ABS_MAX_SHIFT = auto()
    NONE = auto()


def compute_harmonic_mean(
    samples, ln_posterior, model, num_slices: int = 0, shift=Shifting.MEAN_SHIFT
):
    """Compute harmonic mean evidence estimate from samples and log_e posterior values.

    Args:

        samples (np.ndarray[n_chains, n_samples, n_dim]): Samples from posterior distribution.

        ln_posterior (np.ndarray[n_chains, n_samples]): Log_e posterior values at samples.

        model (Model): An instance of a posterior model class that has been fitted.

        shift (Shifting | None): What shifting method to use to avoid over/underflow during
            computation. Selected from enumerate class or None. Defaults to MEAN_SHIFT.

    Returns:
        (float): Estimate of evidence (log).
        (float): Estimate of variance of evidence (log).
        (float): Estimate of variance of the variance of evidence (log).
    """
    X = samples
    Y = ln_posterior
    ln_ratio_per_chain = _compute_harmonic_ratio(
        X,
        Y,
        model,
        num_slices=num_slices,
    )
    shift_value = _determine_shift(ln_ratio_per_chain, shift)
    ln_ratio_per_chain += shift_value

    mask = _create_mask(ln_ratio_per_chain)

    ln_ratio_per_chain = np.ma.masked_array(ln_ratio_per_chain, mask=mask)

    n_samples_per_chain = np.sum(~mask, axis=1)

    # Compute log inverse evidence statistics
    ln_evidence_inv = _compute_ln_evidence_inv(ln_ratio_per_chain, n_samples_per_chain)
    ln_evidence_inv_var = _compute_ln_evidence_inv_var(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )
    ln_evidence_inv_var_var = _compute_ln_evidence_inv_var_var(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )

    # shift back before returning
    return (
        ln_evidence_inv - shift_value,
        ln_evidence_inv_var - 2 * shift_value,
        ln_evidence_inv_var_var - 4 * shift_value,
    )


def evidence_from_ln_inverse(
    ln_evidence_inv: float, ln_evidence_inv_var: float
) -> tuple[float, float]:
    """Compute evidence from the inverse evidence.

    Returns:

        (double, double): Tuple containing the following.

            - evidence (double): Estimate of evidence.

            - evidence_std (double): Estimate of standard deviation of
              evidence.

    Raises:

        ValueError: if inverse evidence or its variance overflows.
    """

    evidence_inv = _exponentiate(ln_evidence_inv)
    evidence_inv_var = _exponentiate(ln_evidence_inv_var)

    if np.isinf(np.nan_to_num(evidence_inv, nan=np.inf)) or np.isinf(
        np.nan_to_num(evidence_inv_var, nan=np.inf)
    ):
        raise ValueError(
            "Evidence is too large to represent in non-log space. Use log-space values instead."
        )

    common_factor = 1.0 + evidence_inv_var / (evidence_inv**2)

    evidence = common_factor / evidence_inv

    evidence_std = np.sqrt(evidence_inv_var) / (evidence_inv**2)

    return (evidence, evidence_std)


def ln_evidence_from_ln_inverse(
    ln_evidence_inv: float, ln_evidence_inv_var: float
) -> tuple[float, float]:
    """Compute log_e of evidence from the inverse evidence.

    Returns:

        (double, double): Tuple containing the following.

            - ln_evidence (double): Estimate of log_e of evidence.

            - ln_evidence_std (double): Estimate of log_e of standard
                deviation of evidence.

    """

    ln_x = ln_evidence_inv_var - 2.0 * ln_evidence_inv
    x = _exponentiate(ln_x)
    ln_evidence = np.log(1.0 + x) - ln_evidence_inv
    ln_evidence_std = 0.5 * ln_evidence_inv_var - 2.0 * ln_evidence_inv

    return (ln_evidence, ln_evidence_std)


def _determine_shift(ln_ratio: np.ndarray, shift: Shifting) -> float:
    """Determine shift value for log_e posterior values to aid numerical stability.

    Args:

        ln_ratio (np.ndarray[n_samples]): Log_e posterior difference values.

        shift (Shifting): What shifting method to use to avoid over/underflow during
            computation. Selected from enumerate class.
    """

    match shift:
        case Shifting.MAX_SHIFT:
            return float(-np.nanmax(ln_ratio))
        case Shifting.MEAN_SHIFT:
            return float(-np.nanmean(ln_ratio))
        case Shifting.MIN_SHIFT:
            return float(-np.nanmin(ln_ratio))
        case Shifting.ABS_MAX_SHIFT:
            return float(-ln_ratio[np.nanargmax(np.abs(ln_ratio))])
        case Shifting.NONE | _:
            return 0.0


def _compute_harmonic_ratio(
    X,
    Y,
    model,
    num_slices: int = 0,
) -> np.ndarray:
    """Compute the log_e ratio values needed for harmonic mean estimator.

    ln_ratio = ln phi(x) - ln p(x)

    where phi(x) is the normalised importance density (model) and p(x) is the unnormalised
    posterior density.

    Args:
        X (np.ndarray[n_chains, n_samples_per_chain, n_dim]): Samples from posterior distribution.

        Y (np.ndarray[n_chains, n_samples_per_chain]): Log_e posterior values at samples.

        model (Model): An instance of an importance sampling model class that has been fitted.

        num_slices (int): Number of slices to use when computing predictions in batches.
            If 0, no slicing is performed.

    """
    n_chains, n_samples_per_chain, n_dim = X.shape
    X = X.reshape(-1, n_dim)
    Y = Y.reshape(-1)

    if num_slices:
        # Number of rows in each slice
        slice_size = X.shape[0] // num_slices
        ln_pred_list = []

        # Calculate ln_pred in row-wise slices
        for i in range(num_slices):
            start_row = i * slice_size
            end_row = (i + 1) * slice_size if i < num_slices - 1 else X.shape[0]
            X_slice = X[start_row:end_row]

            # Predict for each row slice and append result
            ln_pred_slice = model.predict(X_slice)
            ln_pred_list.append(ln_pred_slice)

        # Concatenate all row slice predictions
        ln_pred = np.concatenate(ln_pred_list, axis=0)
    else:
        ln_pred = np.array(model.predict(X))

    ln_ratio = ln_pred - Y
    ln_ratio[np.isinf(ln_ratio)] = np.nan

    return ln_ratio.reshape((n_chains, n_samples_per_chain))


def _compute_n_effective(n_samples_per_chain: np.ndarray) -> float:
    """Compute effective number of samples from total samples and per-chain samples.

    Args:

        n_samples_per_chain (np.ndarray[n_chains]): Number of samples per chain.

    Returns:
        (float): Effective number of samples.
    """
    return np.sum(n_samples_per_chain) ** 2 / np.sum(n_samples_per_chain**2)


def _compute_ln_evidence_inv_per_chain(
    ln_ratio_per_chain: np.ndarray,
    n_samples_per_chain: np.ndarray,
) -> np.ndarray:
    """Compute log_e inverse evidence per chain.

    Args:

        ln_ratio_per_chain (np.ndarray[n_chains, n_samples_per_chain]): Log_e ratio values
            per chain.

        n_samples_per_chain (np.ndarray[n_chains]): Number of samples per chain.

    Returns:
        (np.ndarray[n_chains]): Log_e inverse evidence per chain.
    """
    return sp.logsumexp(ln_ratio_per_chain, axis=1) - np.log(n_samples_per_chain)


def _compute_ln_evidence_inv(
    ln_ratio_per_chain: np.ndarray, n_samples_per_chain: np.ndarray
) -> float:
    """Compute log_e inverse evidence from log_e ratio values.

    Args:

        ln_ratio_per_chain (np.ndarray[n_chains, n_samples_per_chain]): Log_e ratio values
            per chain.
        n_samples (int): Total number of samples across all chains.
    Returns:

        (float): Log_e inverse evidence.
    """
    n_samples = n_samples_per_chain.sum()
    return sp.logsumexp(ln_ratio_per_chain) - np.log(n_samples)


def _compute_ln_evidence_inv_var(
    ln_ratio_per_chain: np.ndarray,
    n_samples_per_chain: np.ndarray,
) -> float:
    """Compute log_e variance of inverse evidence."""

    z_i = _evidence_inv_per_chain_diff(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )
    z_i *= n_samples_per_chain ** (0.5)

    n_samples = n_samples_per_chain.sum()
    n_effective = _compute_n_effective(n_samples_per_chain)

    return sp.logsumexp(2 * np.log(z_i)) - np.log(n_samples) - np.log(n_effective - 1)


def _compute_ln_evidence_inv_var_var(
    ln_ratio_per_chain: np.ndarray,
    n_samples_per_chain: np.ndarray,
) -> float:
    """Compute log_e variance of the variance of inverse evidence."""

    n_effective = _compute_n_effective(n_samples_per_chain)
    ln_evidence_inv_var = _compute_ln_evidence_inv_var(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )
    kur = _compute_kurtosis(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )

    return (
        2.0 * ln_evidence_inv_var
        - np.log(n_effective)
        + np.log(kur - 1.0 + 2.0 / (n_effective - 1.0))
    )


def _ln_ratio_per_chain_to_evidence_inv(
    ln_ratio_per_chain: np.ndarray,
    n_samples_per_chain: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Helper function to convert ln_ratio_per_chain to evidence_inv and evidence_inv_per_chain.

    Needed for variance calculations.
    """

    ln_evidence_inv_per_chain = _compute_ln_evidence_inv_per_chain(
        ln_ratio_per_chain, n_samples_per_chain
    )
    evidence_inv_per_chain = _exponentiate(ln_evidence_inv_per_chain)

    ln_evidence_inv = _compute_ln_evidence_inv(ln_ratio_per_chain, n_samples_per_chain)
    evidence_inv = float(_exponentiate(ln_evidence_inv))

    return evidence_inv, evidence_inv_per_chain


def _compute_kurtosis(
    ln_ratio_per_chain: np.ndarray,
    n_samples_per_chain: np.ndarray,
) -> float:
    """Helper function to compute kurtosis of ln_ratio_per_chain values."""

    y_i = _evidence_inv_per_chain_diff(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )
    y_i *= n_samples_per_chain ** (0.25)

    n_samples = n_samples_per_chain.sum()
    n_effective = _compute_n_effective(n_samples_per_chain)

    ln_evidence_inv_var = _compute_ln_evidence_inv_var(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )

    kur_ln = (
        sp.logsumexp(4.0 * np.log(y_i))
        - np.log(n_samples)
        - 2.0 * ln_evidence_inv_var
        - 2.0 * np.log(n_effective)
    )
    return float(_exponentiate(kur_ln))


def _evidence_inv_per_chain_diff(
    ln_ratio_per_chain: np.ndarray,
    n_samples_per_chain: np.ndarray,
) -> np.ndarray:
    """Compute

        rhohat_i - rhohat

    where rhohat is the inverse evidence estimate and rhohat_i is the per-chain estimate.
    """
    evidence_inv, evidence_inv_per_chain = _ln_ratio_per_chain_to_evidence_inv(
        ln_ratio_per_chain,
        n_samples_per_chain,
    )

    # Difference of each chain's estimate from overall estimate
    # Absolute value is safe because we only take even powers later
    # Clip small values to zero to avoid numerical issues when taking the log
    y_i = np.abs(evidence_inv_per_chain - evidence_inv)
    tol = 1e-15
    y_i[y_i < tol] = 0.0
    return y_i


def _create_mask(ln_ratio_per_chain: np.ndarray) -> np.ndarray:
    """Create mask for invalid log_e ratio values.

    Args:
        ln_ratio_per_chain (np.ndarray[n_chains, n_samples_per_chain]): Log_e ratio values
            per chain.
    Returns:
        (np.ndarray[n_chains, n_samples_per_chain]): Mask array where True indicates invalid
            values.
    """
    mask = np.zeros_like(ln_ratio_per_chain, dtype=bool)
    mask[np.isnan(ln_ratio_per_chain)] = True
    mask[np.isinf(ln_ratio_per_chain)] = True
    return mask


def _exponentiate(ln_values: np.ndarray | float) -> np.ndarray:
    """Exponentiate log_e values, warning if values are too large.

    Args:
        ln_values (np.ndarray | float): Log_e values to exponentiate.
    Returns:
        (np.ndarray): Exponentiated values.
    """
    max_ln_value = 700.0  # Approximate threshold for float64
    if np.any(ln_values > max_ln_value):
        warn(
            "Some log_e values are large and may cause overflow during exponentiation.",
            stacklevel=2,
        )
    return np.exp(ln_values)
