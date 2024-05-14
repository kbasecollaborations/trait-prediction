import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf
from pydantic import ValidationError

from trait_prediction.pipeline import Config


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
