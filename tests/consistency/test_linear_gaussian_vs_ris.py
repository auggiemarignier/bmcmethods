"""Compare the RIS and linear Gaussian results.

Test case is that of a single ND standard normal distribution.
This is already normalised so the expected evidence is 1.0.
"""

import numpy as np
from harmonic.model import RealNVPModel
from linear_gaussian.lg import calc_Z
from ris.estimate import compute_harmonic_mean, evidence_from_ln_inverse
from scipy.stats import multivariate_normal


def calculate_z_lg(nd: int) -> float:
    mus = [np.zeros(nd)]
    Cs = [np.eye(nd)]
    As = [np.eye(nd)]

    Z = calc_Z(mus, Cs, As)
    return Z


def calculate_z_ris(nd: int) -> tuple[float, float]:
    mn = multivariate_normal(mean=np.zeros(nd), cov=np.eye(nd))
    train_samples = mn.rvs(size=5000)
    inference_samples = mn.rvs(size=5000)
    inference_ln_prob = mn.logpdf(inference_samples)

    model = RealNVPModel(nd)
    model.fit(train_samples, epochs=10)

    # very annoying reshaping imposed by harmonic requiring 2D for fitting and 3D for inference
    ln_ev_inv, ln_ev_inv_std, _ = compute_harmonic_mean(
        inference_samples.reshape(2, -1, nd),
        inference_ln_prob.reshape(2, -1),
        model,
    )

    Z, Z_std = evidence_from_ln_inverse(ln_ev_inv, ln_ev_inv_std)
    return Z, Z_std


def test_linear_gaussian_vs_ris():
    nd = 2
    Z_lg = calculate_z_lg(nd)
    Z_ris, Z_ris_std = calculate_z_ris(nd)
    np.testing.assert_allclose(Z_lg, 1.0, rtol=1e-5)
    np.testing.assert_allclose(Z_ris, Z_lg, rtol=0.01)
