"""Tests for linear_gaussian/lg.py."""

import numpy as np
from linear_gaussian import calc_Z
from linear_gaussian.lg import _calc_c, _calc_S, _calc_sT, _calc_y


def test_calc_c_returns_scalar():
    """Scalar c is a scalar value."""

    mus = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    C_invs = [np.eye(2), np.eye(2)]
    c = _calc_c(mus, C_invs)
    assert isinstance(c, float)


def test_calc_sT_returns_correct_shape():
    """Vector sT is of length n where n is the number of parameters."""
    mus = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
    C_invs = [np.eye(2), np.eye(2)]
    As = [np.eye(2), np.eye(2)]
    sT = _calc_sT(mus, C_invs, As)
    assert sT.shape == (2,)


def test_calc_S_returns_correct_shape():
    """S is a matrix of shape (n, n) where n is the number of parameters."""
    C_invs = [np.eye(2), np.eye(2)]
    As = [np.eye(2), np.eye(2)]
    S = _calc_S(C_invs, As)
    assert S.shape == (2, 2)


def test_calc_y_returns_correct_shape():
    """Vector y is of length n where n is the number of parameters."""
    S = np.eye(2)
    s = np.array([1.0, 2.0])
    y = _calc_y(S, s)
    assert y.shape == (2,)


class TestCalcZ:
    """Tests for the main calc_Z function."""

    def test_single_factor_identities_everywhere(self):
        """Single factor, Identity transformations.  Expect Z=1."""
        mus = [np.array([1.0, 2.0])]
        Cs = [np.eye(2)]
        As = [np.eye(2)]
        Z = calc_Z(mus, Cs, As)
        np.testing.assert_array_equal(Z, 1.0)

    def test_single_factor_invertible_transformations(self):
        """
        Single factor, Invertible transformations.

        Expect Z=det(A)^-1
        """
        mus = [np.array([1.0, 2.0])]
        Cs = [np.eye(2)]
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        As = [A]
        Z = calc_Z(mus, Cs, As)
        np.testing.assert_array_equal(Z, 1.0 / np.linalg.det(A))

    def test_two_factors_handcrafted(self):
        """Hand-crafted two-factor test case.

        A1 = [[1, 0, 0]]
        A2 = [[0, 1, 0]
            [0, 0, 1]]
        mu1 = [[0]]
        mu2 = [[0]
            [0]]
        C1 = [[4]]
        C2 = [[1, 0]
            [0, 1]]

        => n = 3, n1 = 1, n2 = 2

        => Expected Z = 1
        """
        A1 = np.array([[1, 0, 0]])
        A2 = np.array([[0, 1, 0], [0, 0, 1]])
        mu1 = np.array([0])
        mu2 = np.array([0, 0])
        C1 = np.array([[4]])
        C2 = np.array([[1, 0], [0, 1]])

        np.testing.assert_array_equal(1.0, calc_Z([mu1, mu2], [C1, C2], [A1, A2]))

    def test_two_factors_dimensions_dont_cancel(self):
        r"""Two factors where every matrix is I and every mu is 1s.

        => expected $Z = 2^{-(n+1)/2}\\pi^{-1/2} exp(n^3-1)$
        """
        n = 3
        mu1 = np.ones(n)
        mu2 = np.ones(n)
        C1 = np.eye(n)
        C2 = np.eye(n)
        A1 = np.eye(n)
        A2 = np.eye(n)

        expected_Z = (4 * np.pi) ** (-n / 2)
        np.testing.assert_array_almost_equal(
            expected_Z, calc_Z([mu1, mu2], [C1, C2], [A1, A2])
        )
