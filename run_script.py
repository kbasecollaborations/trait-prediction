#!/usr/bin/env python

import random

import hydra
import numpy
from omegaconf import DictConfig, OmegaConf

from trait_prediction.pipeline import Config


@hydra.main(config_path="configs/", config_name="default")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config = Config(**cfg_dict)
    # Set random seed
    numpy.random.seed(config.random_state)
    random.seed(config.random_state)
