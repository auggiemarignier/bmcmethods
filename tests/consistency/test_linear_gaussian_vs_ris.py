"""Compare the RIS and linear Gaussian results.

A linear-Gaussian model with:
  - Prior:      X ~ N(0, I_nd)
  - Likelihood: d | X ~ N(X, I_nd)   (A = I, noise = N(0, I))
  - Data:       d = 0

The marginal (evidence) is:
  p(d=0) = N(0; 0, 2*I_nd) = (4*pi)^{-nd/2}

The posterior is:
  p(X | d=0) = N(X; 0, 0.5*I_nd)

We compute the evidence analytically with calc_log_evidence and estimate it
numerically with RIS, then verify they agree.

For RIS, the harmonic mean estimator uses:
  - Samples drawn from the normalised posterior N(0, 0.5*I_nd)
  - Log of the *unnormalised* posterior  log[p(d|X) p(X)]
    = log N(d; X, I) + log N(X; 0, I)
"""

import numpy as np
from harmonic.model import RealNVPModel
from linear_gaussian import GaussianComponent, calc_log_evidence
from ris.estimate import compute_harmonic_mean, evidence_from_ln_inverse
from scipy.stats import multivariate_normal


def calculate_log_z_lg(nd: int, d: np.ndarray) -> float:
    """Compute log-evidence analytically using the linear-Gaussian formula."""
    inferred = [GaussianComponent(A=np.eye(nd), mu=np.zeros(nd), C=np.eye(nd))]
    nuisance = [GaussianComponent(A=np.eye(nd), mu=np.zeros(nd), C=np.eye(nd))]
    return calc_log_evidence(d, inferred, nuisance)


def calculate_z_ris(nd: int, d: np.ndarray) -> tuple[float, float]:
    """Estimate evidence via RIS harmonic mean estimator."""
    # Posterior is N(0, 0.5*I)
    posterior = multivariate_normal(mean=np.zeros(nd), cov=0.5 * np.eye(nd))

    train_samples = posterior.rvs(size=5000)
    inference_samples = posterior.rvs(size=5000)

    # Log of the *unnormalised* posterior: log p(d|X) + log p(X)
    likelihood = multivariate_normal(mean=np.zeros(nd), cov=np.eye(nd))
    prior = multivariate_normal(mean=np.zeros(nd), cov=np.eye(nd))
    inference_ln_prob = likelihood.logpdf(inference_samples - d) + prior.logpdf(
        inference_samples
    )

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
    d = np.zeros(nd)
    log_Z_lg = calculate_log_z_lg(nd, d)
    Z_lg = np.exp(log_Z_lg)
    Z_ris, Z_ris_std = calculate_z_ris(nd, d)
    np.testing.assert_allclose(Z_lg, (4 * np.pi) ** (-nd / 2), rtol=1e-5)
    np.testing.assert_allclose(Z_ris, Z_lg, rtol=0.01)
