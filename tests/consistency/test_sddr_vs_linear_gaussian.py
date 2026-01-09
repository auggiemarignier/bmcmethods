"""Check the consistency between SDDR and linear Gaussian models.

Check that we get the same Bayes Factor using the Savage-Dickey Density Ratio (SDDR)
and the analytical expression for the Bayes Factor in the case of linear Gaussian models.

SDDR only applies for nested models.
Our supermodel is a zero-mean Gaussian with 5 parameters and identity covariance.
The submodel is a zero-mean Gaussian with 3 parameters and identity covariance, nested within the supermodel where the last two parameters are fixed at zero.


Let N_N(x) denote the N-dimensional normal distribution with mean 0 and identity covariance.
The supermodel likelihood is L_super(m) = N_5(m).
Breaking m into the common parameters theta (first 3) and the additional parameters nu (last 2), the supermodel likelihood can be written as:
L_super(theta, nu) = N_3(theta) * N_2(nu).
The prior for the supermodel is taken to be the same as the likelihood:
pi_super(m) = N_5(m) = N_3(theta) * N_2(nu).
So the full supermodel posterior is:
p_super(theta, nu | data) ∝ L_super(theta, nu) * pi_super(theta, nu) = N_3(theta)^2 * N_2(nu)^2.

The marginalised posterior for nu is:
p_super(nu | data) ∝ ∫ p_super(theta, nu | data) dtheta = N_2(nu)^2 * ∫ N_3(theta)^2 dtheta

The submodel likelihood is L_sub(theta) = L_super(theta, nu=0) = N_3(theta) * N_2(0).
The prior for the submodel is pi_sub(theta) = N_3(theta).
So the full submodel posterior is:
p_sub(theta | data) ∝ L_sub(theta) * pi_sub(theta) = N_3(theta)^2 * N_2(0).
i.e. Z_sub = N_2(0) * ∫ N_3(theta)^2 dtheta.

"""

import numpy as np
from linear_gaussian import calc_Z
from scipy.stats import multivariate_normal
from sddr.sddr import sddr


def calculate_bf_lg() -> float:
    """Calculate Bayes Factor using linear Gaussian model evidence."""
    super_mus = [np.zeros(5)] * 2
    super_Cs = [np.eye(5)] * 2
    super_As = [np.eye(5)] * 2
    Z_super = calc_Z(super_mus, super_Cs, super_As)

    sub_mus = [np.zeros(3)] * 2
    sub_Cs = [np.eye(3)] * 2
    sub_As = [np.eye(3)] * 2

    nu = np.zeros(2)
    Z_sub = calc_Z(sub_mus, sub_Cs, sub_As) * multivariate_normal.pdf(
        nu, mean=np.zeros(2), cov=np.eye(2)
    )

    bf_lg = Z_sub / Z_super
    return bf_lg


def calculate_bf_sddr() -> float:
    """Calculate Bayes Factor using SDDR.

    The marginalised posterior and prior can both be written down, so no need for sampling in this case.
    """

    class marginalised_posterior:
        def predict(self, x: np.ndarray) -> np.ndarray:
            posterior_normalisation = (2 * np.pi**0.5) ** -5
            marginalised_posterior_normalisation = (2 * np.pi**0.5) ** -3

            mv = multivariate_normal(
                mean=np.zeros(2),
                cov=np.eye(2),
            )
            return np.log(
                marginalised_posterior_normalisation
                / posterior_normalisation
                * mv.pdf(x) ** 2
            )

    prior_marginal = multivariate_normal(mean=np.zeros(2), cov=np.eye(2))

    nu = np.zeros(2)
    return np.exp(
        sddr(
            marginalised_posterior(),
            prior_marginal.logpdf,
            nu,
        )
    )


def test_sddr_vs_linear_gaussian() -> None:
    """Test that the Bayes Factor from SDDR matches that from linear Gaussian model.

    Analytically, the Bayes Factor is 2.
    """
    bf_lg = calculate_bf_lg()
    bf_sddr = calculate_bf_sddr()
    np.testing.assert_allclose(bf_sddr, bf_lg)
    np.testing.assert_allclose(bf_lg, 2.0)
