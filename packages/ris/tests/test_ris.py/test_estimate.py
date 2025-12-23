import unittest.mock as mock

import harmonic.model as md
import numpy as np
import pytest

from ris.estimate import (
    Shifting,
    _compute_harmonic_ratio,
    _compute_ln_evidence_inv,
    _compute_ln_evidence_inv_var,
    _compute_n_effective,
    _determine_shift,
    compute_harmonic_mean,
    evidence_from_ln_inverse,
    ln_evidence_from_ln_inverse,
    _compute_kurtosis,
)


@pytest.fixture
def posterior_samples() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic posterior samples and unnormalised log-posterior values for testing.

    These come from a 2D standard normal distribution, which has an evidence of 2pi.
    """
    rng = np.random.default_rng(seed=42)
    nchains = 200
    nsamples = 100
    ndim = 2
    X = rng.standard_normal((nchains, nsamples, ndim))
    Y = -0.5 * np.sum(X**2, axis=2)
    return X, Y


class TestShifting:
    def test_determine_shift_mean(self):
        x = np.array([-10.0, -5.0, np.nan, 5.0, 10.0])
        shift_value = _determine_shift(x, Shifting.MEAN_SHIFT)
        assert shift_value == pytest.approx(0.0)

    def test_determine_shift_max(self):
        x = np.array([-10.0, -5.0, 0.0, 5.0, np.nan])
        shift_value = _determine_shift(x, Shifting.MAX_SHIFT)
        assert shift_value == pytest.approx(-5.0)

    def test_determine_shift_min(self):
        x = np.array([np.nan, -5.0, 0.0, 5.0, 10.0])
        shift_value = _determine_shift(x, Shifting.MIN_SHIFT)
        assert shift_value == pytest.approx(5.0)

    def test_determine_shift_abs_max(self):
        x = np.array([-10.0, -5.0, 0.0, 5.0, np.nan])
        shift_value = _determine_shift(x, Shifting.ABS_MAX_SHIFT)
        assert shift_value == pytest.approx(10.0)


@pytest.mark.parametrize(
    "model",
    [
        md.RealNVPModel(2),
        md.RealNVPModel(2, standardize=True),
        md.RQSplineModel(2),
        md.RQSplineModel(2, standardize=True),
    ],
)
@pytest.mark.parametrize("num_slices", [0, 2, 10])
def test_compute_harmonic_mean(
    posterior_samples: tuple[np.ndarray, np.ndarray], model, num_slices: int
):
    X, Y = posterior_samples

    model.fit(X.reshape(-1, X.shape[-1]), epochs=5)

    ln_evidence_inv, ln_evidence_inv_var, ln_evidence_inv_var_var = (
        compute_harmonic_mean(X, Y, model, num_slices=num_slices)
    )
    expected_ln_evidence_inv = np.log(1 / (2 * np.pi))
    assert np.exp(ln_evidence_inv) == pytest.approx(
        np.exp(expected_ln_evidence_inv), rel=0.01
    )

    # NOTE
    # The following checks for variance and variance of variance
    # were skipped for flow models in the original harmonic tests.
    # They passed for the legacy HyperSphere model.
    # I do not know where the expected values come from.
    #
    # assert np.exp(ln_evidence_inv_var) == pytest.approx(1.164628268e-07)
    # assert np.exp(ln_evidence_inv_var_var) ** 0.5 == pytest.approx(1.142786462e-08)


def test_evidence_from_ln_inverse_zero_variance() -> None:
    """In this test, we're assuming we can estimate the evidence of a 2D standard normal distribution perfectly, i.e. 2pi with no variance.

    Probably not the best test as it means we don't actually test the delta correction.
    """
    ln_evidence_inv = np.log(1 / (2 * np.pi))
    ln_evidence_inv_var = -np.inf
    expected_evidence = 2 * np.pi
    estimated_evidence, estimated_evidence_std = evidence_from_ln_inverse(
        ln_evidence_inv, ln_evidence_inv_var
    )
    assert estimated_evidence == pytest.approx(expected_evidence)
    assert estimated_evidence_std == pytest.approx(0.0)


def test_evidence_from_ln_inverse_unity_variance() -> None:
    """In this test, we're assuming we can estimate the evidence of a 2D standard normal distribution with a variance of 1.

    This gives a delta correction factor of (1 + 1 / (1/two_pi)^2)
    => expected evidence = two_pi**3 * (two_pi^-2 + 1)

    The estimated stddev is sqrt(variance / evidence_inv**2)
    => expected_stddev = two_pi**2
    """
    two_pi = 2 * np.pi
    ln_evidence_inv = np.log(1 / (two_pi))
    ln_evidence_inv_var = 0.0
    expected_evidence = two_pi**3 * (two_pi**-2 + 1)
    estimated_evidence, estimated_evidence_std = evidence_from_ln_inverse(
        ln_evidence_inv, ln_evidence_inv_var
    )
    assert estimated_evidence == pytest.approx(expected_evidence)
    assert estimated_evidence_std == pytest.approx(two_pi**2)


def test_ln_evidence_from_ln_inverse() -> None:
    """Test that ln_evidence_from_ln_inverse is consistent with evidence_from_ln_inverse."""
    ln_evidence_inv = np.log(1 / (2 * np.pi))
    ln_evidence_inv_var = 1.0

    estimated_evidence, estimated_evidence_std = evidence_from_ln_inverse(
        ln_evidence_inv, ln_evidence_inv_var
    )
    estimated_ln_evidence, estimated_ln_evidence_std = ln_evidence_from_ln_inverse(
        ln_evidence_inv, ln_evidence_inv_var
    )

    assert np.log(estimated_evidence) == pytest.approx(estimated_ln_evidence)
    assert np.log(estimated_evidence_std) == pytest.approx(estimated_ln_evidence_std)


@pytest.mark.parametrize("num_slices", [0, 2, 10])
def test__compute_harmonic_ratio(
    posterior_samples: tuple[np.ndarray, np.ndarray], num_slices: int
) -> None:
    X, Y = posterior_samples

    class DummyModel:
        def predict(self, x: np.ndarray) -> np.ndarray:
            """Predict log_e density of standard normal distribution.

            Args:
                x (np.ndarray[n_samples, n_dim]): Input samples.
            Returns:
                (np.ndarray[n_samples]): Log_e density values at input samples.
            """
            return -0.5 * np.sum(x**2, axis=1) - np.log(2 * np.pi)

    model = DummyModel()
    ln_ratio = _compute_harmonic_ratio(
        X,
        Y,
        model,
        num_slices=num_slices,
    )

    np.testing.assert_allclose(
        ln_ratio,
        np.full_like(Y, -np.log(2 * np.pi)),  # model is perfect and normalised
    )


def test__compute_n_effective():
    n_samples_per_chain = np.array([300, 400])
    n_effective = _compute_n_effective(n_samples_per_chain)
    assert n_effective == pytest.approx(49 / 25)


def test__compute_ln_evidence_inv(
    posterior_samples: tuple[np.ndarray, np.ndarray],
) -> None:
    """With a perfect normalised model, each chain has the same ln_ratio of -ln(2pi).

    So the mean of these is also -ln(2pi), which is the expected ln_evidence_inv.
    """
    X, Y = posterior_samples
    ln_ratio_per_chain = np.full_like(
        Y, -np.log(2 * np.pi)
    )  # model is perfect and normalised
    n_samples_per_chain = np.full(X.shape[0], X.shape[1])

    estimated_ln_evidence_inv = _compute_ln_evidence_inv(
        ln_ratio_per_chain, n_samples_per_chain
    )
    expected_ln_evidence_inv = -np.log(2 * np.pi)
    assert estimated_ln_evidence_inv == pytest.approx(expected_ln_evidence_inv)


def test__compute_ln_evidence_inv_per_chain(
    posterior_samples: tuple[np.ndarray, np.ndarray],
) -> None:
    """With a perfect normalised model, each chain has the same ln_ratio of -ln(2pi).

    So each chain's ln_evidence_inv is also -ln(2pi).
    """
    X, Y = posterior_samples
    ln_ratio_per_chain = np.full_like(
        Y, -np.log(2 * np.pi)
    )  # model is perfect and normalised
    n_samples_per_chain = np.full(X.shape[0], X.shape[1])

    estimated_ln_evidence_inv_per_chain = _compute_ln_evidence_inv(
        ln_ratio_per_chain, n_samples_per_chain
    )
    expected_ln_evidence_inv_per_chain = np.full(X.shape[0], -np.log(2 * np.pi))
    np.testing.assert_allclose(
        estimated_ln_evidence_inv_per_chain,
        expected_ln_evidence_inv_per_chain,
    )


def test__compute_ln_evidence_inv_var_perfect(
    posterior_samples: tuple[np.ndarray, np.ndarray],
) -> None:
    """With a perfect normalised model, each chain has the same ln_ratio of -ln(2pi) i.e. zero variance."""
    X, Y = posterior_samples
    ln_ratio_per_chain = np.full_like(
        Y, -np.log(2 * np.pi)
    )  # model is perfect and normalised
    n_samples_per_chain = np.full(X.shape[0], X.shape[1])

    estimated_ln_evidence_inv_var = _compute_ln_evidence_inv_var(
        ln_ratio_per_chain, n_samples_per_chain
    )
    expected_ln_evidence_inv_var = -np.inf  # zero variance
    assert estimated_ln_evidence_inv_var == pytest.approx(expected_ln_evidence_inv_var)


def test__compute_ln_evidence_inv_var_nonzero(
    posterior_samples: tuple[np.ndarray, np.ndarray],
) -> None:
    """To force non-zero variance, patch _ln_ratio_per_chain_to_evidence_inv to return different evidence_inv_per_chain values.

    Specifically, we want each chain to have evidence_inv_per_chain values of [1 + 1/2pi] and evidence_inv to be 1 / 2pi.
    This gives (evidence_inv_per_chain - evidence_inv) = 1 for each chain.

    Therefore the expected variance is 1 /(N_eff -1)
    """
    X, Y = posterior_samples
    ln_ratio_per_chain = np.zeros_like(Y)
    n_samples_per_chain = np.full(X.shape[0], X.shape[1])

    with mock.patch(
        "ris.estimate._ln_ratio_per_chain_to_evidence_inv",
        return_value=(1 / (2 * np.pi), np.full(X.shape[0], 1 + 1 / (2 * np.pi))),
    ):
        estimated_ln_evidence_inv_var = _compute_ln_evidence_inv_var(
            ln_ratio_per_chain,
            n_samples_per_chain,
            # The patch overrides the internal uses of these arguments, so their exact values don't matter
        )
        expected_ln_evidence_inv_var = np.log(
            1 / (_compute_n_effective(n_samples_per_chain) - 1)
        )
        assert estimated_ln_evidence_inv_var == pytest.approx(
            expected_ln_evidence_inv_var
        )


def test__compute_kurtosis(
    posterior_samples: tuple[np.ndarray, np.ndarray],
) -> None:
    """To force non-zero kurtosis, patch _ln_ratio_per_chain_to_evidence_inv to return different evidence_inv_per_chain values.

    Specifically, we want each chain to have evidence_inv_per_chain values of [1 + 1/2pi] and evidence_inv to be 1 / 2pi.
    This gives (evidence_inv_per_chain - evidence_inv) = 1 for each chain.

    Therefore the expected kurtosis is 1 / [(N_eff)^2 var^2]
    """
    X, Y = posterior_samples
    ln_ratio_per_chain = np.zeros_like(Y)
    n_samples_per_chain = np.full(X.shape[0], X.shape[1])

    with mock.patch(
        "ris.estimate._ln_ratio_per_chain_to_evidence_inv",
        return_value=(1 / (2 * np.pi), np.full(X.shape[0], 1 + 1 / (2 * np.pi))),
    ):
        estimated_kurtosis = _compute_kurtosis(
            ln_ratio_per_chain,
            n_samples_per_chain,
            # The patch overrides the internal uses of these arguments, so their exact values don't matter
        )
        expected_kurtosis = 1 / (
            _compute_n_effective(n_samples_per_chain) ** 2
            * np.exp(
                _compute_ln_evidence_inv_var(
                    ln_ratio_per_chain,
                    n_samples_per_chain,
                )
            )
            ** 2
        )
        assert estimated_kurtosis == pytest.approx(expected_kurtosis)
