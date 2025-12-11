"""Calculate the evidence for a linear Gaussian model."""

import numpy as np
from numpy.typing import NDArray

type NDArrayFloat = NDArray[np.float64]
type ListArray = list[NDArrayFloat]


def _validate_mus(mus: ListArray) -> None:
    if not isinstance(mus, list):
        raise TypeError("mus must be a list of arrays")
    for mu in mus:
        if not isinstance(mu, np.ndarray):
            raise TypeError("each element of mus must be a numpy array")
        if mu.ndim != 1:
            raise ValueError("each element of mus must be a 1D array")


def _validate_Cs(Cs: ListArray) -> None:
    if not isinstance(Cs, list):
        raise TypeError("Cs must be a list of arrays")
    for C in Cs:
        if not isinstance(C, np.ndarray):
            raise TypeError("each element of Cs must be a numpy array")
        if C.ndim != 2:
            raise ValueError("each element of Cs must be a 2D array")
        if C.shape[0] != C.shape[1]:
            raise ValueError("each element of Cs must be a square matrix")


def _validate_As(As: ListArray) -> None:
    if not isinstance(As, list):
        raise TypeError("As must be a list of arrays")
    for A in As:
        if not isinstance(A, np.ndarray):
            raise TypeError("each element of As must be a numpy array")
        if A.ndim != 2:
            raise ValueError("each element of As must be a 2D array")
        if A.shape[1] != As[0].shape[1]:
            raise ValueError("all elements of As must have the same number of columns")


def _validate_inputs(mus: ListArray, Cs: ListArray, As: ListArray) -> None:
    _validate_mus(mus)
    _validate_Cs(Cs)
    _validate_As(As)
    n = len(mus)
    if len(Cs) != n:
        raise ValueError("mus and Cs must have the same length")
    if len(As) != n:
        raise ValueError("mus and As must have the same length")
    for mu, C, A in zip(mus, Cs, As):
        d = mu.shape[0]
        if C.shape[0] != d:
            raise ValueError("mus and Cs have incompatible dimensions")
        if A.shape[0] != d:
            raise ValueError("mus and As have incompatible dimensions")


def _calc_c(mus: ListArray, C_invs: ListArray) -> float:
    # needs to be a loop because of different sizes so no vectorization possible
    return float(np.sum([mu.T @ C_inv @ mu for mu, C_inv in zip(mus, C_invs)]))


def _calc_sT(mus: ListArray, C_invs: ListArray, As: ListArray) -> NDArrayFloat:
    return np.sum([mu.T @ C_inv @ A for mu, C_inv, A in zip(mus, C_invs, As)], axis=0)


def _calc_S(C_invs: ListArray, As: ListArray) -> NDArrayFloat:
    return np.sum([A.T @ C_inv @ A for C_inv, A in zip(C_invs, As)], axis=0)


def _calc_y(S: NDArrayFloat, s: NDArrayFloat) -> NDArrayFloat:
    return np.linalg.inv(S).T @ s


def calc_Z(mus: ListArray, Cs: ListArray, As: ListArray) -> float:
    """Calculate the model evidence Z for a linear Gaussian model.

    Parameters
    ----------
    mus : list of ndarray
        List of mean vectors for each factor.
    Cs : list of ndarray
        List of covariance matrices for each factor.
    As : list of ndarray
        List of transformation matrices for each factor.

    Returns
    -------
    Z : float
        The model evidence.
    """

    _validate_inputs(mus, Cs, As)

    C_invs = [np.array(np.linalg.inv(C)) for C in Cs]

    c = _calc_c(mus, C_invs)
    sT = _calc_sT(mus, C_invs, As)
    S = _calc_S(C_invs, As)
    y = _calc_y(S, sT.T)

    yTSy = y.T @ S @ y
    det_S_inv = np.linalg.det(S) ** -0.5
    if np.isinf(det_S_inv):
        raise ValueError("S matrix is singular, cannot compute its inverse.")
    det_Cs_inv = np.prod([np.linalg.det(C) ** -0.5 for C in Cs])

    nx = As[0].shape[1]
    ns = [mu.shape[0] for mu in mus]
    prefactor = (2 * np.pi) ** (0.5 * (nx - sum(ns))) * det_S_inv * det_Cs_inv
    exponential = np.exp(-0.5 * (c - yTSy))

    return prefactor * exponential
