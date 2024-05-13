"""Module that defines the Pipeline class"""

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from tqdm import tqdm

from ..logging import logger
from ..main import DataSet, Feature, FeatureInput, Phenotype, PhenotypeInput
from ..training import Predictor
from .config import Config
from .experiment import Experiment


@dataclass
class TaskData:
    """The TaskData class defines the data for a task.

    Attributes
    ----------
    phenotype : Phenotype
        The phenotype data.
    feature : Feature
        The feature data.
    experiment : Experiment
        The experiment object.
    config : Config
        The configuration object.
    make_classifier : Callable[[int, list[str] | None], Any]
        The function to create a classifier.
    output_dir : Path
        The output directory.
    random_state : int
        The random state.
    """

    phenotype: Phenotype
    feature: Feature
    experiment: Experiment
    config: Config
    make_classifier: Callable[[int, list[str] | None], Any]
    output_dir: Path
    random_state: int

    def get_metadata(self) -> dict:
        """Get the metadata for the task.

        Returns
        -------
        dict
            Task metadata.
        """
        metadata = {
            "phenotype": {
                "name": self.phenotype.pindex.name,
                "category": self.phenotype.pindex.category,
            },
            "feature": {
                "name": self.feature.findex.name,
                "ftype": self.feature.findex.ftype,
                "dtype": self.feature.findex.dtype,
            },
            "experiment": self.experiment.experiment_dir,
            "config": self.config.model_dump(),
            "random_state": self.random_state,
        }
        return metadata


class PredictionPipeline:
    def __init__(
        self,
        config_path: Path,
        pinputs: list[PhenotypeInput],
        finputs: list[FeatureInput],
        make_classifier: Callable[[int, list[str] | None], Any],
        output_dir: Path,
        n_cpus: int,
        random_state: int,
    ):
        self.config = Config.load_config(config_path)
        self.dataset = DataSet.read_data(pinputs, finputs)
        self.make_classifier = make_classifier
        self.output_dir = output_dir
        self.n_cpus = n_cpus
        self.random_state = random_state
        self._initialize_experiment()

    def _initialize_experiment(self):
        """Initialize the experiment."""
        existing_dirs = [d.name for d in self.output_dir.iterdir() if d.is_dir()]
        self.experiment = Experiment.initialize(existing_dirs, "_")
        metadata = {
            "config": self.config.model_dump(),
            "n_cpus": self.n_cpus,
            "random_state": self.random_state,
            "phenotypes": [
                {"name": p.pindex.name, "category": p.pindex.category}
                for p in self.dataset.phenotype_set
            ],
            "features": [
                {
                    "name": f.findex.name,
                    "ftype": f.findex.ftype,
                    "dtype": f.findex.dtype,
                }
                for f in self.dataset.feature_set
            ],
        }
        self.experiment.log_metadata(metadata)

    @staticmethod
    def is_xdata_good(feature_data: pd.DataFrame, config: Config) -> bool:
        """Check if the feature data is good for training.

        Parameters
        ----------
        feature_data : pd.DataFrame
            The data frame containing the feature data.

        Returns
        -------
        bool
            True if the data is good for training, otherwise False.
        """
        if feature_data.shape[0] <= config.phenotype_sample_size_threshold:
            return False
        if feature_data.shape[1] <= 1:
            return False
        return True

    @staticmethod
    def is_ydata_good(phenotype_data: pd.Series, config: Config) -> bool:
        """Check if the phenotype data is good for training.

        Parameters
        ----------
        phenotype_data : pd.Series
            The series containing the phenotype data.

        Returns
        -------
        bool
            True if the data is good for training, otherwise False.
        """
        # Skip if phenotype has only one class
        if len(phenotype_data.unique()) == 1:
            return False
        if phenotype_data.shape[0] <= config.phenotype_sample_size_threshold:
            return False
        if (
            phenotype_data.value_counts().min()
            <= config.minor_class_sample_size_threshold
        ):
            return False
        return True

    @staticmethod
    def preprocess_feature_data(
        feature_data: pd.DataFrame, ftype: str, config: Config
    ) -> tuple[pd.DataFrame, list[str], dict[str, list[str]]]:
        """Preprocess the feature data.

        Parameters
        ----------
        feature_data : pd.DataFrame
            The data frame containing the feature data.
        ftype : str
            The type of the feature data.
        config : Config
            The configuration object.

        Returns
        -------
        tuple[pd.DataFrame, list[str], dict[str, list[str]]]
            The preprocessed feature data, the features with low variance, and the correlated features.
        """
        # Variance filtering
        # TODO: Can we support multiple variance thresholds to avoid this if-else block?
        if ftype == "binary":
            feature_data, low_var_features = Feature.remove_features_with_low_variance(
                feature_data, config.variance_threshold
            )
        else:
            low_var_features = []
        # Correlation filtering
        if config.correlation_threshold is not None:
            # TODO: Is it possible to avoid hardcoding the method here?
            n_features = feature_data.shape[1]
            if n_features <= 40_000:
                corr_method = "numpy"
            else:
                corr_method = "numba"
            feature_data, corr_group_dict = (
                Feature.remove_features_with_high_correlation(
                    feature_data, config.correlation_threshold, method=corr_method
                )
            )
        else:
            corr_group_dict = {}
        return feature_data, low_var_features, corr_group_dict

    @staticmethod
    def select_features(
        feature_data: pd.DataFrame, phenotype_data: pd.Series, config: Config
    ) -> tuple[pd.DataFrame, list[str], pd.DataFrame | None]:
        """Feature selection or reduction is applied to the feature data based on the config.

        Parameters
        ----------
        feature_data : pd.DataFrame
            The data frame containing the feature data.
        phenotype_data : pd.Series
            The series containing the phenotype data.
        config : Config
            The configuration object.

        Returns
        -------
        tuple[pd.DataFrame, list[str], pd.DataFrame | None]
            The selected feature data, the low score features, and the components dataframe.
        """
        if config.score_function is not None:
            feature_data, low_score_features = Feature.feature_selection_kbest(
                feature_data, phenotype_data, config.score_function
            )
            components_df = None
        elif config.reduction_function is not None:
            feature_data, components_df = Feature.feature_dimensionality_reduction(
                feature_data, config.reduction_function, config.n_feature_reduction
            )
            low_score_features = []
        else:
            raise ValueError("No feature selection or reduction method specified")
        return feature_data, low_score_features, components_df

    @staticmethod
    def _run_task(task_data: TaskData):
        phenotype = task_data.phenotype
        feature = task_data.feature
        config = task_data.config
        experiment = task_data.experiment
        make_classifier = task_data.make_classifier
        random_state = task_data.random_state
        experiment_result = experiment.create_result()
        metadata = task_data.get_metadata()
        # Log the metadata
        experiment_result.log_metadata(metadata)
        ftype = feature.findex.ftype
        feature_data = feature.feature_data
        phenotype_data = phenotype.phenotype_data
        # Check if the data is good for training
        if not PredictionPipeline.is_xdata_good(feature_data, config):
            return None
        if not PredictionPipeline.is_ydata_good(phenotype_data, config):
            return None
        # Preprocess the feature data
        feature_data, low_var_features, corr_group_dict = (
            PredictionPipeline.preprocess_feature_data(feature_data, ftype, config)
        )
        # Select features
        feature_data, low_score_features, components_df = (
            PredictionPipeline.select_features(feature_data, phenotype_data, config)
        )
        phenotype_train = Phenotype(phenotype_data, phenotype.pindex)
        feature_train = Feature(feature_data, feature.findex)
        # Log the preprocessing data
        experiment_result.log_preprocessing_data(
            low_var_features, corr_group_dict, low_score_features
        )
        # Create the predictor
        if ftype == "binary":
            categorical_feature_names = []
            for col in phenotype_data.columns:
                if str(phenotype_data[col].dtype).startswith("uint"):
                    categorical_feature_names.append(col)
        else:
            categorical_feature_names = None
        classifier = make_classifier(random_state, categorical_feature_names)
        predictor = Predictor(phenotype_train, feature_train, classifier, random_state)
        # Split the data into training and testing sets
        if config.cross_validation:
            predictor.split_data_cv(n_splits=config.n_splits, stratify=True)
            score = predictor.get_score(kind="CV", n_jobs=1, scoring=config.scoring)
        else:
            predictor.split_data(
                sampling_type=config.sampling_type,
                test_size=config.test_size,
                imbalance_correction=config.imbalance_correction,
                stratify=True,
            )
            score = predictor.get_score(kind="test", n_jobs=1, scoring=config.scoring)
        # Log the training data
        experiment_result.log_data(predictor)
        # Log the metrics
        experiment_result.log_metrics(score)
        # Log the models
        if config.log_models:
            experiment_result.log_models(score)
        # Log the plots
        experiment_result.log_plots(score, feature_data, config)

    def run(self):
        """Run the pipeline."""
        tasks: list[TaskData] = []
        for feature_raw in self.dataset.feature_set:
            for phenotype_raw in self.dataset.phenotype_set:
                phenotype, feature = self.dataset.get_data(
                    phenotype_raw.pindex, feature_raw.findex
                )
                task = TaskData(
                    phenotype=phenotype,
                    feature=feature,
                    experiment=self.experiment,
                    config=self.config,
                    make_classifier=self.make_classifier,
                    output_dir=self.output_dir,
                    random_state=self.random_state,
                )
                tasks.append(task)
        with mp.Pool(self.n_cpus) as pool:
            results = []
            # TODO: Replace tqdm with mp.Value & logger to track progress
            for result in tqdm(pool.imap(self._run_task, tasks), total=len(tasks)):
                results.append(result)
