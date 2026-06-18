"""Run configuration loading and sweeps (``pipeline.config``)."""

import pytest
from pydantic import ValidationError

from trait_prediction.pipeline import Config, ConfigSet


def test_load_valid_config(config_path):
    config = Config.load_config(config_path / "default.yaml")
    assert config.random_state == 42
    assert config.classifier.name == "catboost"


def test_load_invalid_config_raises(config_path):
    with pytest.raises(ValidationError):
        Config.load_config(config_path / "bad_config.yaml")


def test_configset_sweep(config_path):
    base_config = Config.load_config(config_path / "default.yaml")
    config_set = ConfigSet.create_configset(base_config, config_path / "config_set.yaml")
    # 2 selection_functions x 2 random_states = 4 configs
    assert len(config_set) == 4
