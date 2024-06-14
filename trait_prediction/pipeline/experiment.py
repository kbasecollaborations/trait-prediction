"""Module that defines the Experiment class"""

import gzip
import json
import random
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from ..training import Predictor, Score
from ..visualization.feature_importances import plot_shap_summary
from .config import Config, ConfigSet

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NAMES = json.loads((DATA_DIR / "names.json").read_text())


@dataclass
class File:
    """The File class represents a file.

    Attributes
    ----------
    name : str
        The name of the file.
    path : Path
        The path to the file.
    """

    name: str
    path: Path


class ExperimentResult:
    """The ExperimentResult class represents the result of a run of an experiment.

    Parameters
    ----------
    run_dir : Path
        The directory where the run is stored.

    Attributes
    ----------
    run_dir : Path
        The directory where the run is stored.
    metadata : dict
        The metadata of the run.
    files : dict[str, list[File]
        The files of the run.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        if not self.run_dir.exists():
            self.run_dir.mkdir(parents=True)
        self.metadata = {}
        self.files: dict[str, list[File]] = {}

    def __hash__(self):
        return hash(self.run_dir)

    @staticmethod
    def generate_run_id(existing_dirs: list[str], length: int = 12) -> str:
        """Generate a unique run ID.

        Parameters
        ----------
        existing_dirs : list[str]
            The list of existing directories.
        length : int
            The length of the run ID.
            Default is 12.

        Returns
        -------
        str
            The run ID.
        """
        alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
        if existing_dirs:
            run_dir = existing_dirs[0]
            while run_dir in existing_dirs:
                run_dir = "".join(random.choices(alphabet, k=length))
        else:
            run_dir = "".join(random.choices(alphabet, k=length))
        return run_dir

    @classmethod
    def initialize(cls, base_dir: Path, length: int = 12) -> "ExperimentResult":
        """Initialize a new ExperimentResult.

        Parameters
        ----------
        base_dir : Path
            The base directory.
        length : int
            The length of the run ID.

        Returns
        -------
        "ExperimentResult"
            The ExperimentResult object.
        """
        existing_dirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
        run_dir = base_dir / cls.generate_run_id(existing_dirs, length)
        run_dir.mkdir(parents=True, exist_ok=True)
        return cls(run_dir)

    def parse(self):
        """Parse the contents run directory."""
        with open(self.run_dir / "metadata.yaml") as fid:
            self.metadata = yaml.safe_load(fid)
        # Read data file paths
        data_dir = self.run_dir / "data"
        data_files = []
        for data_file in data_dir.iterdir():
            data_files.append(File(data_file.name, data_file))
        # Read model file paths
        model_dir = self.run_dir / "models"
        model_files = []
        for model_file in model_dir.iterdir():
            model_files.append(File(model_file.name, model_file))
        # Read metrics
        metrics_dir = self.run_dir / "metrics"
        metrics_files = []
        for metrics_file in metrics_dir.iterdir():
            metrics_files.append(File(metrics_file.name, metrics_file))
        # Read plots
        plots_dir = self.run_dir / "plots"
        plots_files = []
        for plot_file in plots_dir.iterdir():
            plots_files.append(File(plot_file.name, plot_file))
        files = {
            "data": data_files,
            "models": model_files,
            "metrics": metrics_files,
            "plots": plots_files,
        }
        self.files = files

    def log_metadata(self, metadata: dict) -> None:
        """Log the metadata to a file."""
        self.metadata = metadata
        with open(self.run_dir / "metadata.yaml", "w") as fid:
            yaml.safe_dump(self.metadata, fid)

    def log_preprocessing_data(
        self,
        low_var_features: list[str],
        correlated_features_dict: dict[str, list[str]],
        low_score_features: list[str],
        components_df: pd.DataFrame | None,
        feature_data: pd.DataFrame,
    ) -> None:
        """Log the preprocessing data.

        Parameters
        ----------
        low_var_features : list[str]
            The features with low variance that were removed.
        correlated_features_dict : dict[str, list[str]]
            The features with high correlation that were removed.
        low_score_features : list[str]
            The features with low score_func score that were removed.
        components_df : pd.DataFrame | None
            The components dataframe.
        feature_data : pd.DataFrame
            The features dataframe.
        """
        output_dir = self.run_dir / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "low_var_features_list.txt", "w") as fid:
            fid.write("\n".join(low_var_features))
        with gzip.open(output_dir / "corr_features_map.json.gz", "wt") as gzfile:
            json.dump(correlated_features_dict, gzfile)
        with open(output_dir / "low_score_features_list.txt", "w") as fid:
            fid.write("\n".join(low_score_features))
        if components_df is not None:
            components_df.to_csv(output_dir / "components.tsv", sep="\t", index=True)
            feature_data.to_csv(
                output_dir / "features_reduced.tsv", sep="\t", index=True
            )

    def log_data(self, predictor: Predictor) -> None:
        """Log the training data.

        Parameters
        ----------
        predictor : Predictor
            The predictor object.
        """
        output_dir = self.run_dir / "data"
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_data = False
        # save the training data
        if predictor.training_data is not None:
            predictor.training_data.save_indices(output_dir)
            saved_data = True
        # save the cv data
        if predictor.cv_data is not None:
            predictor.cv_data.save_indices(predictor.phenotype, output_dir)
            saved_data = True
        if saved_data is False:
            raise ValueError("Neither Training data nor CV data set for the predictor")

    def log_metrics(self, score: Score) -> None:
        """Save the metrics.

        Parameters
        ----------
        score : Score
            The score object.
        """
        output_dir = self.run_dir / "metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        score.save_scores(output_dir)

    def log_models(self, score: Score, subset: str, metric: str) -> None:
        """Save the models.

        Parameters
        ----------
        score : Score
            The score object.
        subset : str
            The subset of the estimators to save. Either 'all' or 'best'.
        metric : str
            The metric to use for selecting the best estimators.
            Only used if subset is 'best'.
        """
        output_dir = self.run_dir / "models"
        output_dir.mkdir(parents=True, exist_ok=True)
        score.save_estimators(output_dir, subset, metric)

    def log_plots(self, score: Score, X: pd.DataFrame, config: Config) -> None:
        """Save the visualizations.

        Parameters
        ----------
        score : Score
            The score object.
        X : pd.DataFrame
            The entire X data.
        config : Config
            The configuration object.
        """
        output_dir = self.run_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, classifier in enumerate(score.estimators):
            shap_summary_plot_file = str(output_dir / f"shap_summary_plot_{i}.png")
            shap_features_file = str(output_dir / f"shap_features_{i}.csv")
            fname = score.findex.name
            pname = score.pindex.name
            title = f"F={fname}, P={pname}"
            importance_df = plot_shap_summary(
                classifier,
                X,
                config,
                title=title,
                output_file=shap_summary_plot_file,
            )
            importance_df.to_csv(shap_features_file, index=True, sep=",")


class Experiment(Set[ExperimentResult]):
    """The Experiment class represents an experiment.

    Parameters
    ----------
    experiment_dir : Path
        The directory where the experiment is stored.

    Attributes
    ----------
    experiment_dir : Path
        The directory where the experiment is stored.
    metadata : dict
        The metadata of the experiment.
    config : Config
        The configuration of the experiment.
    """

    _names = NAMES

    def __init__(self, experiment_dir: Path, config: Config | None = None):
        self.experiment_dir = experiment_dir
        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)
        self._results = set()
        self._config: Config | None = config
        self.metadata = {}

    def __repr__(self):
        return f"Experiment (n={len(self._results)}, dir={self.experiment_dir})"

    def __len__(self):
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def __contains__(self, item):
        return item in self._results

    def __hash__(self):
        return hash(self.experiment_dir)

    @property
    def config(self) -> Config | None:
        return self._config

    @config.setter
    def config(self, config: Config) -> None:
        self._config = config

    @classmethod
    def generate_experiment_id(cls, existing_dirs: list[str], sep: str) -> str:
        """Generate a unique experiment ID.

        Parameters
        ----------
        existing_dirs : list[str]
            The list of existing directories.
        sep : str
            The separator between the left and right names.

        Returns
        -------
        str
            The experiment ID.
        """
        if existing_dirs:
            experiment_dir = existing_dirs[0]
            while experiment_dir in existing_dirs:
                left_name = random.choice(cls._names["left"])
                right_name = random.choice(cls._names["right"])
                experiment_dir = f"{left_name}{sep}{right_name}"
        else:
            left_name = random.choice(cls._names["left"])
            right_name = random.choice(cls._names["right"])
            experiment_dir = f"{left_name}{sep}{right_name}"
        return experiment_dir

    @classmethod
    def initialize(
        cls,
        base_dir: Path,
        sep: str,
    ) -> "Experiment":
        """Initialize a new Experiment.

        Parameters
        ----------
        base_dir : Path
            The base directory.
        sep : str
            The separator between the left and right names.

        Returns
        -------
        "Experiment"
            The Experiment object.
        """
        existing_dirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
        experiment_dir = base_dir / cls.generate_experiment_id(existing_dirs, sep)
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return cls(experiment_dir)

    def create_result(self) -> ExperimentResult:
        return ExperimentResult.initialize(self.experiment_dir)

    def parse(self):
        """Parse the contents of the experiment directory."""
        with open(self.experiment_dir / "metadata.yaml") as fid:
            self.metadata = yaml.safe_load(fid)
        for run_dir in self.experiment_dir.iterdir():
            run_result = ExperimentResult(run_dir)
            run_result.parse()
            self._results.add(run_result)

    def log_metadata(self, metadata: dict) -> None:
        """Log the metadata to a file."""
        self.metadata = metadata
        with open(self.experiment_dir / "metadata.yaml", "w") as fid:
            yaml.safe_dump(self.metadata, fid)


class ExperimentSet(Set[Experiment]):
    _names = NAMES

    def __init__(self, experimentset_dir: Path):
        self.experimentset_dir = experimentset_dir
        if not self.experimentset_dir.exists():
            self.experimentset_dir.mkdir(parents=True)
        self._experiments = set()
        self.metadata = {}

    def __repr__(self):
        return (
            f"ExperimentSet (n={len(self._experiments)}, dir={self.experimentset_dir})"
        )

    def __len__(self):
        return len(self._experiments)

    def __iter__(self):
        return iter(self._experiments)

    def __contains__(self, item):
        return item in self._experiments

    @classmethod
    def generate_experimentset_id(cls, existing_dirs: list[str], sep: str) -> str:
        """Generate a unique experimentset ID.

        Parameters
        ----------
        existing_dirs : list[str]
            The list of existing directories.
        sep : str
            The separator between the left and right names.

        Returns
        -------
        str
            The experimentset ID.
        """
        if existing_dirs:
            experimentset_dir = existing_dirs[0]
            while experimentset_dir in existing_dirs:
                left_name = random.choice(cls._names["left"])
                middle_name = random.choice(cls._names["left"])
                right_name = random.choice(cls._names["right"])
                experimentset_dir = f"{left_name}{sep}{middle_name}{sep}{right_name}"
        else:
            left_name = random.choice(cls._names["left"])
            middle_name = random.choice(cls._names["left"])
            right_name = random.choice(cls._names["right"])
            experimentset_dir = f"{left_name}{sep}{middle_name}{sep}{right_name}"
        return experimentset_dir

    @classmethod
    def initialize(
        cls,
        base_dir: Path,
        sep: str,
    ) -> "ExperimentSet":
        """Initialize a new ExperimentSet.

        Parameters
        ----------
        base_dir : Path
            The base directory.
        sep : str
            The separator between the left and right names.

        Returns
        -------
        "ExperimentSet"
            The ExperimentSet object.
        """
        existing_dirs = [d.name for d in base_dir.iterdir() if d.is_dir()]
        experimentset_dir = base_dir / cls.generate_experimentset_id(existing_dirs, sep)
        experimentset_dir.mkdir(parents=True, exist_ok=True)
        return cls(experimentset_dir)

    def create_experiment(self) -> Experiment:
        """Create a new experiment."""
        return Experiment.initialize(self.experimentset_dir, sep="_")

    def create_experiments(
        self, config_set: ConfigSet, common_metadata: dict
    ) -> set[Experiment]:
        """Create a set of experiments.

        Parameters
        ----------
        config_set : ConfigSet
            The configuration set.
        common_metadata : dict
            The common metadata.

        Returns
        -------
        set[Experiment]
            The set of experiments.
        """
        experiments = set()
        for config in config_set.configs:
            experiment = self.create_experiment()
            experiment.config = config
            metadata = {
                **common_metadata,
                "config": config.model_dump(),
            }
            experiment.log_metadata(metadata)
            self._experiments.add(experiment)
            experiments.add(experiment)
        return experiments

    def parse(self):
        """Parse the contents of the experimentset directory."""
        with open(self.experimentset_dir / "config_set.yaml") as fid:
            self.metadata = yaml.safe_load(fid)
        for experiment_dir in self.experimentset_dir.iterdir():
            experiment = Experiment(experiment_dir)
            experiment.parse()
            self._experiments.add(experiment)

    @staticmethod
    def _serialize_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def log_metadata(self, metadata: dict) -> None:
        """Log the metadata to a file."""
        self.metadata = metadata
        with open(self.experimentset_dir / "metadata.json", "w") as fid:
            json.dump(self.metadata, fid, default=self._serialize_sets)
