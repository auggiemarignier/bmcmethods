"""Compare transC sampling and Reciprocal Importance Sampling."""

import numpy as np
import pytest
from harmonic.model import RealNVPModel
from ris.estimate import compute_harmonic_mean, evidence_from_ln_inverse
from scipy.stats import multivariate_normal
from transc import get_relative_marginal_likelihoods, run_ensemble_resampler


def calculate_z_ris(nd: int) -> tuple[float, float]:
    mn = multivariate_normal(mean=np.zeros(nd), cov=np.eye(nd))
    train_samples = mn.rvs(size=5000)
    inference_samples = mn.rvs(size=5000)
    inference_ln_prob = mn.logpdf(inference_samples)

    # Unnormalise
    normalisation = np.log((2 * np.pi) ** (nd / 2))
    inference_ln_prob += normalisation

    model = RealNVPModel(nd, standardize=True, temperature=0.9)
    model.fit(train_samples, epochs=10)

    # very annoying reshaping imposed by harmonic requiring 2D for fitting and 3D for inference
    ln_ev_inv, ln_ev_inv_std, _ = compute_harmonic_mean(
        inference_samples.reshape(200, -1, nd),
        inference_ln_prob.reshape(200, -1),
        model,
    )

    Z, Z_std = evidence_from_ln_inverse(ln_ev_inv, ln_ev_inv_std)
    return Z, Z_std


def calculate_B_ris(nd1: int, nd2: int) -> float:
    Z1, _ = calculate_z_ris(nd1)
    Z2, _ = calculate_z_ris(nd2)

    B = Z1 / Z2
    return B


def calculate_B_transc(nd1: int, nd2: int) -> float:
    mv = multivariate_normal(mean=np.zeros(nd1), cov=np.eye(nd1))
    state1_samples = mv.rvs(size=5000)
    state1_ln_prob = mv.logpdf(state1_samples)
    mv = multivariate_normal(mean=np.zeros(nd2), cov=np.eye(nd2))
    state2_samples = mv.rvs(size=5000)
    state2_ln_prob = mv.logpdf(state2_samples)

    # Unnormalise
    normalisation1 = np.log((2 * np.pi) ** (nd1 / 2))
    normalisation2 = np.log((2 * np.pi) ** (nd2 / 2))

    resampled_chains = run_ensemble_resampler(
        n_walkers=50,
        n_steps=2000,
        n_states=2,
        n_dims=[nd1, nd2],
        log_posterior_ens=[
            state1_ln_prob + normalisation1,
            state2_ln_prob + normalisation2,
        ],
        log_pseudo_prior_ens=[state1_ln_prob, state2_ln_prob],
    )
    rmls = get_relative_marginal_likelihoods(
        resampled_chains,
        500,  # burn-in
        15,  # thinning
        walker_average="median",
    )
    B = rmls[0] / rmls[1]
    return B


@pytest.mark.flaky(reruns=3)
def test_transc_vs_ris():
    nd1 = 5
    nd2 = 2
    B_ris = calculate_B_ris(nd1, nd2)
    B_transc = calculate_B_transc(nd1, nd2)
    np.testing.assert_allclose(B_transc, B_ris, rtol=0.1)
