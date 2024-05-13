"""Module that defines the Config class"""

import pathlib
from typing import Literal

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


class Config(BaseModel):
    """The Config class defines the configuration for the pipeline.

    Attributes
    ----------
    score_function : The scoring function for feature selection. One of 'f_classif', 'chi2', 'mutual_info_classif'
    n_feature_selection : The number of features to select
    reduction_function : The reduction function for feature dimensionality reduction. One of 'PCA', 'NMF'
    n_feature_reduction : The number of features to reduce to
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
    """

    model_config = ConfigDict(extra="forbid")
    score_function: Literal["f_classif", "chi2", "mutual_info_classif"]
    n_feature_selection: int = Field(gt=0)
    reduction_function: Literal["PCA", "NMF"]
    n_feature_reduction: int = Field(gt=0)
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
    scoring: list[str]
    log_models: bool

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
