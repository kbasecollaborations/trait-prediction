"""Module that defines the Config class"""

import pathlib

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Config(BaseModel):
    """The Config class defines the configuration for the pipeline.

    Attributes
    ----------
    score_functions : The scoring functions for feature selection
    n_feature_selection : The number of features to select
    reduction_functions : The reduction functions for feature dimensionality reduction
    n_feature_reduction : The number of features to reduce to
    random_seed: The random seed to use
    variance_treshold : The variance threshold to use for feature reduction
    correlation_threshold : The correlation threshold to use for feature reduction
    imbalanced : Whether to perform imbalanced sampling
    test_size : The size of the test set
    n_splits : The number of splits to use for cross-validation
    phenotype_sample_size_threshold : The threshold for the minimum number of samples for a phenotype
    minor_class_sample_size_threshold : The threshold for the minimum number of samples for a minor class
    shap_max_display : The maximum number of features to display in SHAP plots
    scoring : The scoring metrics to calculate
    count_features : The features to treat as count features
    float_features : The features to treat as float features
    """

    model_config = ConfigDict(extra="forbid")
    score_functions: list[str]
    n_feature_selection: int = Field(gt=0)
    reduction_functions: list[str]
    n_feature_reduction: int = Field(gt=0)
    random_seed: int
    variance_treshold: float = Field(ge=0, le=1)
    correlation_threshold: float | None = Field(None, ge=0, le=1)
    imbalanced: bool
    test_size: float = Field(gt=0, lt=1)
    n_splits: int = Field(gt=0)
    phenotype_sample_size_threshold: int = Field(gt=0)
    minor_class_sample_size_threshold: int = Field(gt=0)
    shap_max_display: int = Field(gt=0)
    scoring: list[str]
    count_features: list[str]
    float_features: list[str]

    @classmethod
    def load_config(cls, config_path: pathlib.Path) -> "Config":
        """Load the configuration from a file.

        Parameters
        ----------
        config_path : pathlib.Path
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
