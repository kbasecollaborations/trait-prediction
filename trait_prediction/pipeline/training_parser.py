"""Module that defines the TrainingParser class"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from .experiment import ExperimentSet


def flatten_dict(nested_dict: dict, parent_key: str = "", sep: str = "_") -> dict:
    """Flatten a nested dictionary by concatenating keys and moving inner dictionary values

    Parameters
    ----------
    nested_dict : dict
        The dictionary to flatten.
    parent_key : str, optional
        The base string to prepend to keys during concatenation. Defaults to ''.
    sep : str, optional
        The separator to use between concatenated keys. Defaults to '_'.

    Returns
    -------
    dict
        A new dictionary with flattened keys and innermost values.
    """
    items = []
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, "-".join(v)))
        else:
            items.append((new_key, v))
    return dict(items)


class TrainingParser:
    def __init__(self, experimentset_dir: Path) -> None:
        self.experimentset = ExperimentSet(experimentset_dir)
        self.parse()

    def _parse_metadata(self, metadata: dict) -> dict:
        """Parse metadata from the training results."""
        feature_metadata = flatten_dict(metadata["feature"], parent_key="feature")
        phenotype_metadata = flatten_dict(metadata["phenotype"], parent_key="phenotype")
        config_metadata = flatten_dict(metadata["config"])
        parsed_metadata = {**feature_metadata, **phenotype_metadata, **config_metadata}
        return parsed_metadata

    def _parse_scores(self, files: list) -> pd.DataFrame:
        """Parse scores from the training results."""
        if len(files) > 1:
            raise ValueError("Multiple score files found")
        score_file = files[0].path
        scores_df = pd.read_csv(score_file, index_col=0)
        scores_df["fold_num"] = scores_df.index
        return scores_df

    def _parse_importances(self, files: list) -> pd.DataFrame:
        """Parse feature importances from the training results."""

        def read_importances(file_path: Path, fold_num: int) -> pd.DataFrame:
            df = pd.read_csv(file_path)
            df.columns = ["feature", "importance"]
            df["fold_num"] = fold_num
            return df

        importances = []
        filtered_files = [
            f for f in sorted(files, key=lambda x: x.name) if f.path.suffix == ".csv"
        ]
        for fold_num, file in enumerate(filtered_files):
            file_path = file.path
            importances.append(read_importances(file_path, fold_num))
        importances_df = pd.concat(importances, axis=0, ignore_index=True)
        return importances_df

    def parse(self) -> None:
        """Parse the training results from the experiment set."""
        self.experimentset.parse()
        result_scores_df_list = []
        result_importances_df_list = []
        metadata = defaultdict(dict)
        for experiment in self.experimentset.experiments:
            experiment_id = experiment.experiment_dir.stem
            for result in experiment.results:
                result_id = result.run_dir.stem
                result_metadata = self._parse_metadata(result.metadata)
                result_scores_df = self._parse_scores(result.files["metrics"])
                result_importances = self._parse_importances(result.files["plots"])
                result_scores_df["experiment_id"] = experiment_id
                result_scores_df["result_id"] = result_id
                for k, v in result_metadata.items():
                    result_scores_df[k] = v
                    result_importances[k] = v
                result_scores_df_list.append(result_scores_df)
                result_importances_df_list.append(result_importances)
                metadata[experiment_id][result_id] = result_metadata
        self.metadata = metadata
        self.scores = pd.concat(result_scores_df_list, axis=0, ignore_index=True)
        self.importances = pd.concat(
            result_importances_df_list, axis=0, ignore_index=True
        )
