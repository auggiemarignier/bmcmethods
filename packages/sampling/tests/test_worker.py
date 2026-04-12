from multiprocessing import Pool

import numpy as np
import pytest
import sampling._worker as worker_module
from sampling._worker import init_worker, logl, logp


class DummyLikelihood:
    """Simple picklable likelihood with an integer state multiplier."""

    def __init__(self, factor: int) -> None:
        self.state = factor

    @classmethod
    def from_state(cls, state: int) -> "DummyLikelihood":
        return cls(state)

    def __call__(self, model_params: np.ndarray) -> float:
        return float(np.sum(model_params) * self.state)


class DummyPrior:
    """Simple prior object implementing the PriorFunction protocol."""

    def __init__(self, offset: float = 0.0, n: int = 1) -> None:
        self.config_params: list[np.ndarray] = []
        self.n = n
        self.offset = offset

    def __call__(self, model_params: np.ndarray) -> float:
        return float(np.sum(model_params) + self.offset)

    def gradient(self, model_params: np.ndarray) -> np.ndarray:
        return np.zeros_like(model_params)

    def sample(self, num_samples: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(num_samples, self.n))


def test_worker_pool_logl_logp() -> None:
    """Demonstrate initialising workers with `init_worker` and using `logl`/`logp`.

    This spawns a small Pool with the module initializer and maps `logl`/`logp`
    across simple inputs to show that the globals are reconstructed in each
    worker process.
    """

    # Initialise worker processes with DummyLikelihood state=3 and DummyPrior offset=0.5
    with Pool(
        processes=2,
        initializer=init_worker,
        initargs=(DummyLikelihood, 3, DummyPrior(0.5)),
    ) as p:
        inputs = [np.array([1.0]), np.array([2.0])]
        logl_results = p.map(logl, inputs)
        logp_results = p.map(logp, inputs)

    assert logl_results == [3.0, 6.0]
    assert logp_results == [1.5, 2.5]


def test_logl_raises_when_uninitialised() -> None:
    # ensure globals are unset in the main process
    worker_module.WORKER_LIKELIHOOD = None
    with pytest.raises(RuntimeError, match="likelihood"):
        logl(np.array([1.0]))


def test_logp_raises_when_uninitialised() -> None:
    # ensure globals are unset in the main process
    worker_module.WORKER_PRIOR = None
    with pytest.raises(RuntimeError, match="prior"):
        logp(np.array([1.0]))
