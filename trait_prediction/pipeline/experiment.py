"""Module that defines the Experiment class"""

import json
import random
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path

import yaml

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
        run_dir = existing_dirs[0]
        while run_dir in existing_dirs:
            run_dir = "".join(random.choices(alphabet, k=length))
        return run_dir

    @classmethod
    def initialize(
        cls, existing_dirs: list[str], length: int = 12
    ) -> "ExperimentResult":
        """Initialize a new ExperimentResult.

        Parameters
        ----------
        existing_dirs : list[str]
            The list of existing directories.
        length : int
            The length of the run ID.

        Returns
        -------
        "ExperimentResult"
            The ExperimentResult object.
        """
        run_dir = Path(cls.generate_run_id(existing_dirs, length))
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

    def log_metadata(self):
        """Log the metadata to a file."""
        with open(self.run_dir / "metadata.yaml", "w") as fid:
            yaml.safe_dump(self.metadata, fid)

    def log_data(self):
        pass

    def log_metric(self):
        pass

    def log_plot(self):
        pass

    def log_model(self):
        pass


# TODO: Create a class ExperimentSet that contains multiple Experiment objects
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
    """

    _names = NAMES

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        if not self.experiment_dir.exists():
            self.experiment_dir.mkdir(parents=True)
        self._results = set()
        self.metadata = {}

    def __repr__(self):
        return f"Experiment (n={len(self._results)}, dir={self.experiment_dir})"

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
        experiment_dir = existing_dirs[0]
        while experiment_dir in existing_dirs:
            left_name = random.choice(cls._names["left"])
            right_name = random.choice(cls._names["right"])
            experiment_dir = f"{left_name}{sep}{right_name}"
        return experiment_dir

    @classmethod
    def initialize(
        cls,
        existing_dirs: list[str],
        sep: str,
    ) -> "Experiment":
        """Initialize a new Experiment.

        Parameters
        ----------
        existing_dirs : list[str]
            The list of existing directories.
        sep : str
            The separator between the left and right names.

        Returns
        -------
        "Experiment"
            The Experiment object.
        """
        experiment_dir = Path(cls.generate_experiment_id(existing_dirs, sep))
        experiment_dir.mkdir(parents=True, exist_ok=True)
        return cls(experiment_dir)

    def create_result(self) -> ExperimentResult:
        existing_dirs = [
            dir.name for dir in self.experiment_dir.iterdir() if dir.is_dir()
        ]
        return ExperimentResult.initialize(existing_dirs)

    def parse(self):
        """Parse the contents of the experiment directory."""
        with open(self.experiment_dir / "metadata.yaml") as fid:
            self.metadata = yaml.safe_load(fid)
        for run_dir in self.experiment_dir.iterdir():
            run_result = ExperimentResult(run_dir)
            run_result.parse()
            self._results.add(run_result)

    def log_metadata(self):
        """Log the metadata to a file."""
        with open(self.experiment_dir / "metadata.yaml", "w") as fid:
            yaml.safe_dump(self.metadata, fid)
