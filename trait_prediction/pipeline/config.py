"""Module that defines the Config class"""

import json
from collections.abc import Set
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from sklearn.metrics import get_scorer_names
from typing_extensions import Self

SelectionFunctionOpts = Literal[
    "score:f_classif",
    "score:chi2",
    "score:mutual_info_classif",
    "reduction:PCA",
    "reduction:NMF",
]

ClassifierOpts = Literal[
    "catboost",
    "nearest_neighbors",
    "identity",
    "bernoulli",
]


class Config(BaseModel):
    """The Config class defines the configuration for the pipeline.

    Attributes
    ----------
    selection_function : The function to use for feature selection
    n_feature_selection : The number of features to select
    random_state: The random seed to use
    variance_threshold : The variance threshold to use for feature reduction
    correlation_threshold : The correlation threshold to use for feature reduction
    sampling_type : The type of sampling to use for train-test split (random, ooc)
    imbalance_correction : The type of imbalanced sampling to use (oversample, undersample, auto, None)
    test_size : The size of the test set
    cross_validation : Whether to perform cross-validation
    n_splits : The number of splits to use for cross-validation
    phenotype_sample_size_threshold : The threshold for the minimum number of samples for a phenotype
    minor_class_sample_size_threshold : The threshold for the minimum number of samples for a minor class
    shap_max_display : The maximum number of features to display in SHAP plots
    scoring : The scoring metrics to calculate
    log_models : Whether to save the estimators
    classifier : The classifier to use
    classifier_kwargs : The keyword arguments for the classifier
    """

    model_config = ConfigDict(extra="forbid")
    selection_function: SelectionFunctionOpts | None
    n_feature_selection: int = Field(gt=0)
    random_state: int
    variance_threshold: float = Field(ge=0, le=1)
    correlation_threshold: float | None = Field(None, ge=0, le=1)
    sampling_type: Literal["random", "ooc"]
    imbalance_correction: Literal["oversample", "undersample", "auto"] | None
    test_size: float = Field(gt=0, lt=1)
    cross_validation: bool
    n_splits: int = Field(gt=0)
    phenotype_sample_size_threshold: int = Field(gt=0)
    minor_class_sample_size_threshold: int = Field(gt=0)
    shap_max_display: int = Field(gt=0)
    scoring: tuple[str, ...]
    log_models: bool
    classifier: ClassifierOpts
    classifier_kwargs: dict[str, Any]

    def __hash__(self):
        model_dict = self.model_dump()
        model_str = json.dumps(model_dict, sort_keys=True)
        return hash(model_str)

    @field_validator("scoring")
    @classmethod
    def check_scoring(cls, v: list[str]) -> list[str]:
        scorer_names = get_scorer_names()
        if not all(method in scorer_names for method in v):
            raise ValueError("Invalid scoring method")
        return v

    @model_validator(mode="after")
    def check_functions(self) -> Self:
        score_function = self.score_function
        reduction_function = self.reduction_function
        if score_function is not None and reduction_function is not None:
            raise ValueError(
                "Cannot specify both score_function and reduction_function"
            )
        return self

    @property
    def score_function(self) -> str | None:
        if self.selection_function is None:
            return None
        func_type, func_name = self.selection_function.split(":")
        if func_type == "score":
            return func_name
        else:
            return None

    @property
    def reduction_function(self) -> str | None:
        if self.selection_function is None:
            return None
        func_type, func_name = self.selection_function.split(":")
        if func_type == "reduction":
            return func_name
        else:
            return None

    @classmethod
    def load_config(cls, config_path: Path) -> "Config":
        """Load the configuration from a file.

        Parameters
        ----------
        config_path : Path
            The path to the configuration file

        Returns
        -------
        "Config"
            The configuration object
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        try:
            verified_config = cls(**config)
        except ValidationError as e:
            raise e
        return verified_config


class ConfigSet(Set[Config]):
    """The ConfigSet class defines a set of configurations.

    Attributes
    ----------
    configs : The set of configurations
    config_set : The merged configuration set as a dictionary
    """

    def __init__(self, configs: Iterable[Config]) -> None:
        self.configs = set(configs)

    def __len__(self) -> int:
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)

    def __contains__(self, item) -> bool:
        return item in self.configs

    @classmethod
    def create_configset(
        cls, base_config: Config, config_set_path: Path
    ) -> "ConfigSet":
        """Create a configuration set using the base_config and a config_set

        Parameters
        ----------
        base_config : Config
            The base configuration
        config_set_path : Path
            The path to the configuration set file

        Returns
        -------
        "ConfigSet"
            The configuration set object
        """
        configs = set()
        base_config_dict = base_config.model_dump()
        with open(config_set_path, "r") as file:
            config_set = yaml.safe_load(file)
        keys, values = zip(*config_set.items())
        new_config_dicts = [
            dict(zip(keys, combination)) for combination in product(*values)
        ]
        for new_config_dict in new_config_dicts:
            updated_config = {**base_config_dict, **new_config_dict}
            try:
                verified_config = Config(**updated_config)
            except ValidationError as e:
                raise e
            configs.add(verified_config)
        return cls(configs)

    @property
    def config_set(self) -> dict:
        """The merged configuration set"""
        merged_config = dict()
        for config in self.configs:
            for key, value in config.model_dump().items():
                if key not in merged_config:
                    merged_config[key] = set()
                if isinstance(value, dict):
                    new_value = tuple([(k, v) for k, v in value.items()])
                else:
                    new_value = value
                merged_config[key].add(new_value)
        return merged_config

    @classmethod
    def load_configs(cls, config_paths: Iterable[Path]) -> "ConfigSet":
        """Load the configuration set from a list of files.

        Parameters
        ----------
        config_paths : Iterable[Path]
            An iterable of paths to the configuration files

        Returns
        -------
        "ConfigSet"
            The configuration set object
        """
        configs = set()
        for config_path in config_paths:
            with open(config_path, "r") as file:
                config = yaml.safe_load(file)
            try:
                verified_config = cls(**config)
            except ValidationError as e:
                raise e
            configs.add(verified_config)
        return cls(configs)
