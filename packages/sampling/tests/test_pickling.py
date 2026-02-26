import pickle

import numpy as np
from sampling.likelihood import GaussianLikelihood
from sampling.posterior import Posterior
from sampling.priors.component import PriorComponent
from sampling.priors.compound import CompoundPrior
from sampling.priors.gaussian import GaussianPrior, GaussianPriorComponentConfig
from sampling.priors.uniform import UniformPrior, UniformPriorComponentConfig
from sampling.priors.wrapped import (
    WrappedUniformPrior,
)


def forward_fn(model_params: np.ndarray) -> np.ndarray:
    """Simple forward function that returns model parameters as predictions."""
    return np.atleast_2d(model_params)


def forward_fn_grad(model_params: np.ndarray) -> np.ndarray:
    """Gradient of the forward function (identity, so gradient is ones)."""
    model_params = np.atleast_2d(model_params)
    return np.ones_like(model_params)


def test_priors_are_picklable():
    mean = np.zeros(2)
    inv = np.eye(2)
    gp = GaussianPrior(mean=mean, inv_covar=inv)

    lower = np.array([-1.0, -1.0])
    upper = np.array([1.0, 1.0])
    up = UniformPrior(lower_bounds=lower, upper_bounds=upper)

    wup = WrappedUniformPrior(lower_bounds=lower, upper_bounds=upper)

    # Component configs -> PriorComponent -> CompoundPrior
    gconf = GaussianPriorComponentConfig(mean=mean, inv_covar=inv, indices=[0, 1])
    pcomp = gconf.to_prior_component()

    uconf = UniformPriorComponentConfig(
        lower_bounds=lower, upper_bounds=upper, indices=[0, 1]
    )
    ucomp = uconf.to_prior_component()

    cp = CompoundPrior([pcomp, ucomp])

    # Try pickling and unpickling
    for obj in (gp, up, wup, pcomp, ucomp, cp):
        dumped = pickle.dumps(obj)
        loaded = pickle.loads(dumped)
        # basic smoke check: calling on a test vector should not error
        x = np.zeros(2)
        if isinstance(loaded, PriorComponent):
            # call the wrapped prior function on the appropriate subset
            loaded.prior_fn(x[loaded.indices])
        else:
            loaded(x)


def test_likelihood_and_posterior_are_picklable():
    # Build a likelihood using a top-level forward function
    obs = np.zeros(2)
    lik = GaussianLikelihood(
        forward_fn=forward_fn,
        forward_fn_gradient=forward_fn_grad,
        observed_data=obs,
        inv_covar=np.array([1.0]),
    )

    # Use a simple Gaussian prior
    mean = np.zeros(2)
    inv = np.eye(2)
    gp = GaussianPrior(mean=mean, inv_covar=inv)

    post = Posterior(likelihood_fn=lik.__call__, prior_fn=gp.__call__)

    for obj in (lik, post):
        dumped = pickle.dumps(obj)
        loaded = pickle.loads(dumped)
        # smoke: evaluate on a simple input
        x = np.zeros(2)
        loaded(x)
