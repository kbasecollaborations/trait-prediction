import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from trait_prediction.pipeline import Config, ConfigSet


def test_default_config(default_config_path):
    config_path = default_config_path / "default.yaml"
    Config.load_config(config_path)


def test_bad_config(default_config_path):
    config_path = default_config_path / "bad_config.yaml"
    with pytest.raises(ValidationError):
        Config.load_config(config_path)


def test_hydra_load_config(hydra_path):
    config_path = str(hydra_path / "configs")
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name="default")
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)  # type: ignore
    Config(**cfg_dict)


def test_configset_create(default_config_path, default_config):
    config_set_path = default_config_path / "config_set.yaml"
    base_config = default_config
    config_set = ConfigSet.create_configset(base_config, config_set_path)
    assert len(config_set) == 4


def test_configset_configset(default_configset):
    configset = default_configset
    config_set = configset.config_set
    assert len(config_set["selection_function"]) == 2
    assert len(config_set["random_state"]) == 2


def test_configset_classifiers(default_config_path, default_config):
    config_set_path = default_config_path / "config_set_classifiers.yaml"
    base_config = default_config
    config_set = ConfigSet.create_configset(base_config, config_set_path)
    print(config_set.config_set)
    assert len(config_set) == 2
