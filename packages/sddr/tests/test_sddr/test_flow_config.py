import sys

import pytest
from sddr.sddr import FlowModelConfig, RealNVPConfig, RQSplineConfig


def _save_and_remove_harmonic():
    saved = {k: sys.modules[k] for k in list(sys.modules) if k.startswith("harmonic")}
    for k in list(saved):
        del sys.modules[k]
    return saved


def _restore(saved):
    # remove any harmonic entries created during the test
    for k in list(sys.modules):
        if k.startswith("harmonic"):
            del sys.modules[k]
    sys.modules.update(saved)


def test_config_initialisation_does_not_import_harmonic():
    saved = _save_and_remove_harmonic()
    try:
        assert "harmonic" not in sys.modules

        # Instantiating the config should not import harmonic
        _ = RealNVPConfig()
        assert "harmonic" not in sys.modules

        _ = RQSplineConfig()
        assert "harmonic" not in sys.modules
    finally:
        _restore(saved)


@pytest.mark.parametrize("cfg_cls", [RealNVPConfig, RQSplineConfig])
def test_model_cls_imports_harmonic(cfg_cls: type[FlowModelConfig]):
    saved = _save_and_remove_harmonic()
    try:
        cfg = cfg_cls()
        assert "harmonic" not in sys.modules

        # Calling model_cls should import harmonic and harmonic.model
        cls = cfg.model_cls()
        assert "harmonic" in sys.modules
        assert "harmonic.model" in sys.modules
        # sanity-check: returned object is a class
        assert isinstance(cls, type)
    finally:
        _restore(saved)
