"""Module that defines the Pipeline class"""

import multiprocessing as mp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from ..logging import logger
from ..main import (
    DataSet,
    Feature,
    FeatureIndex,
    FeatureInput,
    Phenotype,
    PhenotypeInput,
)
from ..training import Predictor
from .config import Config, ConfigSet
from .experiment import Experiment, ExperimentSet

logger.remove()
logger_file = logger.bind(experiment="main", run="main", file=True)
logger_std = logger.bind(experiment="main", run="main", stdout=True)


class ClassifierType(Protocol):
    def __call__(
        self, random_state: int, categorical_feature_names: list[str] | None, **kwargs
    ) -> Any: ...


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
    classifier_factory : dict[str, ClassifierType]
        The dictionary that contains functions to create a classifier.
    output_dir : Path
        The output directory.
    random_state : int
        The random state.
    n_tasks : int
        The number of tasks.
    """

    phenotype: Phenotype
    feature: Feature
    experiment: Experiment
    config: Config
    classifier_factory: dict[str, ClassifierType]
    output_dir: Path
    random_state: int
    n_tasks: int = 0

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
            "experiment": str(self.experiment.experiment_dir),
            "config": self.config.model_dump(),
            "random_state": self.random_state,
        }
        return metadata


class TrainingPipeline:
    """The class that defines the pipeline for training the trait prediction models.

    Attributes
    ----------
    configset : The configuration set object
    classifier_factory : The dictionary that contains functions to create a classifier
    output_dir : The output directory
    n_cpus : The number of processors to use for training
    random_state : The random state
    experimentset : The experiment set object
    dataset : The dataset object
    """

    def __init__(
        self,
        configset: ConfigSet,
        pinputs: list[PhenotypeInput],
        finputs: list[FeatureInput],
        classifier_factory: dict[str, ClassifierType],
        output_dir: Path,
        n_cpus: int,
        random_state: int,
    ):
        logger_file.enable("trait_prediction")
        self.configset = configset
        self.classifier_factory = classifier_factory
        self.output_dir = output_dir
        self.n_cpus = n_cpus
        self.random_state = random_state
        self.experimentset = ExperimentSet.initialize(self.output_dir, sep="_")
        log_file = self.experimentset.experimentset_dir / "experimentset.log"
        logger_file.add(
            log_file,
            enqueue=True,
            filter=lambda record: "file" in record["extra"],
            mode="w",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | Experiment:{extra[experiment]}, Run:{extra[run]} - {message}",
        )
        logger_std.add(
            sys.stdout,
            enqueue=True,
            filter=lambda record: "stdout" in record["extra"],
            colorize=True,
            diagnose=True,
            backtrace=True,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | "
                "<yellow>Experiment:{extra[experiment]}, Run:{extra[run]}</yellow> - {message}"
            ),
        )
        logger_file.info(
            f"Initialized experimentset at {self.experimentset.experimentset_dir}"
        )
        self.dataset = DataSet.read_data(pinputs, finputs)
        logger_file.info("Loaded dataset")
        self._update_metadata()
        logger_file.info("Updated and logged metadata")
        common_metadata = {
            k: v for k, v in self.experimentset.metadata.items() if k != "configset"
        }
        self.experimentset.create_experiments(self.configset, common_metadata)
        n_experiments = len(self.experimentset)
        logger_file.info(
            f"Created {n_experiments} experiment folders and logged metadata"
        )
        logger_std.info(
            f"Pipeline fully initialized at {self.experimentset.experimentset_dir}"
        )
        for experiment in self.experimentset:
            if experiment.config.classifier not in self.classifier_factory:
                error_str = f"Classifier {experiment.config.classifier} not found in the classifier factory"
                logger_file.error(error_str)
                logger_std.error(error_str)
                raise ValueError(error_str)

    def _update_metadata(self) -> None:
        metadata = {
            "configset": self.configset.config_set,
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
        self.experimentset.log_metadata(metadata)

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
            feature_data, corr_group_dict = (
                Feature.remove_features_with_high_correlation(
                    feature_data, config.correlation_threshold, parallel=False
                )
            )
        else:
            corr_group_dict = {}
        return feature_data, low_var_features, corr_group_dict

    @staticmethod
    def select_features(
        feature_data: pd.DataFrame,
        phenotype_data: pd.Series,
        findex: FeatureIndex,
        config: Config,
    ) -> tuple[pd.DataFrame, FeatureIndex, list[str], pd.DataFrame | None]:
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
            new_findex = FeatureIndex(findex.name, findex.ftype, findex.dtype)
        elif config.reduction_function is not None:
            feature_data, components_df = Feature.feature_dimensionality_reduction(
                feature_data, config.reduction_function, config.n_feature_selection
            )
            new_findex = FeatureIndex(findex.name, "float", "float64")
            low_score_features = []
        else:
            raise ValueError("No feature selection or reduction method specified")
        return feature_data, new_findex, low_score_features, components_df

    @staticmethod
    def _run_task(task_data: TaskData, progress, lock):
        """Run the task.

        Parameters
        ----------
        task_data : TaskData
            The task data.
        """
        experiment = task_data.experiment
        experiment_result = experiment.create_result()
        task_logger = logger_file.bind(
            experiment=experiment.experiment_dir.name,
            run=experiment_result.run_dir.name,
        )
        task_logger_std = logger_std.bind(
            experiment=experiment.experiment_dir.name,
            run=experiment_result.run_dir.name,
        )
        phenotype = task_data.phenotype
        feature = task_data.feature
        config = task_data.config
        classifier_factory = task_data.classifier_factory
        make_classifier = classifier_factory[config.classifier]
        random_state = task_data.random_state
        # Log the metadata
        task_logger.info("Task setup successfully. Logging metadata")
        metadata = task_data.get_metadata()
        experiment_result.log_metadata(metadata)
        ftype = feature.findex.ftype
        feature_data = feature.feature_data
        phenotype_data = phenotype.phenotype_data
        task_logger.info("Preprocessing data")
        # Check if the data is good for training
        if not TrainingPipeline.is_xdata_good(feature_data, config):
            return None
        if not TrainingPipeline.is_ydata_good(phenotype_data, config):
            return None
        task_logger.info("Data is good for training")
        # Preprocess the feature data
        feature_data, low_var_features, corr_group_dict = (
            TrainingPipeline.preprocess_feature_data(feature_data, ftype, config)
        )
        task_logger.info("Feature data preprocessed")
        # Select features
        feature_data, new_findex, low_score_features, components_df = (
            TrainingPipeline.select_features(
                feature_data, phenotype_data, feature.findex, config
            )
        )
        task_logger.info("Features selected")
        phenotype_selected = Phenotype(phenotype_data, phenotype.pindex)
        feature_selected = Feature(feature_data, new_findex)
        # Log the preprocessing data
        experiment_result.log_preprocessing_data(
            low_var_features,
            corr_group_dict,
            low_score_features,
            components_df,
            feature_data,
        )
        task_logger.info("Preprocessing data logged")
        # Create the predictor
        if ftype == "binary":
            categorical_feature_names = []
            for col in feature_data.columns:
                if str(feature_data[col].dtype).startswith("uint"):
                    categorical_feature_names.append(col)
        else:
            categorical_feature_names = None
        classifier = make_classifier(
            random_state, categorical_feature_names, **config.classifier_kwargs
        )
        predictor = Predictor(
            phenotype_selected, feature_selected, classifier, random_state
        )
        task_logger.info("Predictor created")
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
        task_logger.info("Data split and scored")
        # Log the training data
        experiment_result.log_data(predictor)
        task_logger.info("Data logged")
        # Log the metrics
        experiment_result.log_metrics(score)
        task_logger.info("Metrics logged")
        # Log the models
        if config.log_models:
            experiment_result.log_models(score)
            task_logger.info("Models logged")
        # Log the plots
        experiment_result.log_plots(score, feature_data, config)
        task_logger.info("Plots logged")
        lock.acquire()
        progress.value += 1
        task_logger.info(
            f"Completed task. Progress: {progress.value} of {task_data.n_tasks}"
        )
        task_logger_std.info(
            f"Completed task. Progress: {progress.value} of {task_data.n_tasks}",
        )
        lock.release()

    def run(self):
        """Run the pipeline."""
        tasks = []
        logger_file.info("Running the pipeline")
        manager = mp.Manager()
        progress = manager.Value("i", 0)
        lock = manager.Lock()
        for experiment in self.experimentset:
            for feature_raw in self.dataset.feature_set:
                for phenotype_raw in self.dataset.phenotype_set:
                    phenotype, feature = self.dataset.get_data(
                        phenotype_raw.pindex, feature_raw.findex
                    )
                    task = TaskData(
                        phenotype=phenotype,
                        feature=feature,
                        experiment=experiment,
                        config=experiment.config,
                        classifier_factory=self.classifier_factory,
                        output_dir=self.output_dir,
                        random_state=self.random_state,
                    )
                    tasks.append(task)
        for task in tasks:
            task.n_tasks = len(tasks)
        logger_file.info(f"Generated {len(tasks)} tasks")
        logger_std.info(f"Generated {len(tasks)} tasks")
        with mp.Pool(self.n_cpus) as pool:
            pool.starmap(self._run_task, [(task, progress, lock) for task in tasks])
        logger_file.info(
            f"Pipeline completed. Completed tasks: {progress.value} of {len(tasks)}"
        )
