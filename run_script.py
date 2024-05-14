#!/usr/bin/env python

import hydra
from omegaconf import DictConfig, OmegaConf

from trait_prediction.pipeline import Config


@hydra.main(config_path="configs/", config_name="default")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config = Config(**cfg_dict)
