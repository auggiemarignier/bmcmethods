"""Replicating the example in the SDDR notebook in `harmonic`.

https://github.com/astro-informatics/harmonic/blob/main/notebooks/gaussian_sddr_example.ipynb
"""

import harmonic as hm
import numpy as np
from ris.estimate import compute_harmonic_mean, ln_evidence_from_ln_inverse
from sampling.likelihood import GaussianLikelihood
from sampling.priors import GaussianPrior
from sampling.sampling import MCMCConfig, mcmc
from sddr.sddr import sddr

rng = np.random.default_rng(42)


def init_cov(ndim):
    """Initialise random non-diagonal covariance matrix.

    Args:

        ndim: Dimension of Gaussian.

    Returns:

        cov: Covariance matrix of shape (ndim,ndim).

    """

    cov = np.zeros((ndim, ndim))
    diag_cov = np.ones(ndim) + rng.normal(size=ndim) * 0.1
    np.fill_diagonal(cov, diag_cov)
    off_diag_size = 0.5
    for i in range(ndim - 1):
        cov[i, i + 1] = (
            (-1) ** i * off_diag_size * np.sqrt(cov[i, i] * cov[i + 1, i + 1])
        )
        cov[i + 1, i] = cov[i, i + 1]

    return cov


def test_sddr_vs_ris():
    """Just as one big script for now.

    The results of the different methods will not be exactly the same, but should be close enough to 0.9.
    """
    n_params = 5
    n_nested = 2

    means = np.zeros(n_params)
    prior_cov = np.diag(np.ones(n_params) * 2.0**2)
    inv_prior_cov = np.linalg.inv(prior_cov)
    prior = GaussianPrior(mean=means, inv_covar=inv_prior_cov)
    nested_prior = GaussianPrior(
        mean=means[:-n_nested], inv_covar=inv_prior_cov[:-n_nested, :-n_nested]
    )
    marginal_prior = GaussianPrior(
        mean=means[-n_nested:], inv_covar=inv_prior_cov[-n_nested:, -n_nested:]
    )

    ndata = 3
    cov = init_cov(ndata)
    inv_cov = np.linalg.inv(cov)

    theta_truth = np.array([0.0, -0.5, 0.5])
    eta_truth = np.array([1.0, 2.0])

    truth = np.concatenate((theta_truth, eta_truth))

    def mock_simulator(theta):
        theta = np.atleast_2d(theta)
        model_prediction = np.zeros((theta.shape[0], ndata))
        model_prediction[:, 0] = theta[:, 0]
        model_prediction[:, 1] = np.arcsinh(theta[:, 1]) + np.arctan(theta[:, 3])
        model_prediction[:, 2] = np.exp(0.5 * theta[:, 2]) - theta[:, 4]
        return model_prediction

    def nested_mock_simulator(theta):
        theta = np.atleast_2d(theta)
        model_prediction = np.zeros((theta.shape[0], ndata))
        model_prediction[:, 0] = theta[:, 0]
        model_prediction[:, 1] = np.arcsinh(theta[:, 1]) + np.arctan(eta_truth[0])
        model_prediction[:, 2] = np.exp(0.5 * theta[:, 2]) - eta_truth[1]
        return model_prediction

    noiseless_data = mock_simulator(truth)[0]

    ln_likelihood_super = GaussianLikelihood(
        forward=mock_simulator,
        observed_data=noiseless_data,
        inv_covar=inv_cov,
    )
    ln_likelihood_nested = GaussianLikelihood(
        forward=nested_mock_simulator,
        observed_data=noiseless_data,
        inv_covar=inv_cov,
    )

    mcmc_cfg = MCMCConfig(nwalkers=100, nsteps=6000, parallel=False)

    super_samples, super_ln_prob = mcmc(
        n_params,
        ln_likelihood_super,
        prior,
        rng,
        mcmc_cfg,
    )

    nested_samples, nested_ln_prob = mcmc(
        n_params - n_nested,
        ln_likelihood_nested,
        nested_prior,
        rng,
        mcmc_cfg,
    )

    marginalised_samples = super_samples[:, :, -n_nested:].reshape(-1, n_nested)

    histogram_model = hm.model_classical.HistogramModel(ndim=2, nbins=300)
    histogram_model.fit(marginalised_samples)
    hist_log_bf = sddr(histogram_model, marginal_prior, eta_truth)

    print(
        f"The Bayes factor calculated from the SDDR with the classical histogram model is: {hist_log_bf:.4f}"
    )

    flow_model = hm.model.RQSplineModel(ndim_in=2, standardize=True, temperature=1.0)
    flow_model.fit(marginalised_samples, epochs=10)
    flow_log_bf = sddr(flow_model, marginal_prior, eta_truth)
    print(
        f"The Bayes factor calculated from the SDDR with the normalising flow model is: {flow_log_bf:.4f}"
    )

    assert np.isclose(flow_log_bf, 0.9, atol=0.1), (
        "SDDR with flow model did not reproduce expected value."
    )

    temperature = 0.9
    standardize = True
    epochs_num = 10

    super_model = hm.model.RQSplineModel(
        n_params, standardize=standardize, temperature=temperature
    )
    super_model.fit(
        super_samples.reshape(-1, n_params), epochs=epochs_num, verbose=True
    )
    nested_model = hm.model.RQSplineModel(
        n_params - n_nested, standardize=standardize, temperature=temperature
    )
    nested_model.fit(
        nested_samples.reshape(-1, n_params - n_nested), epochs=epochs_num, verbose=True
    )
    super_ris_z = compute_harmonic_mean(
        super_samples.swapaxes(0, 1),
        super_ln_prob.swapaxes(0, 1),
        super_model,
    )
    nested_ris_z = compute_harmonic_mean(
        nested_samples.swapaxes(0, 1),
        nested_ln_prob.swapaxes(0, 1),
        nested_model,
    )
    ris_log_bf = (
        ln_evidence_from_ln_inverse(*nested_ris_z[:2])[0]
        - ln_evidence_from_ln_inverse(*super_ris_z[:2])[0]
    )
    print(f"The log evidence ratio from RIS is: {ris_log_bf:.4f}")

    assert np.isclose(ris_log_bf, 0.9, atol=0.1), (
        "Bayes factor from RIS did not reproduce expected value."
    )
