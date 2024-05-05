"""Module that defines the Pipeline class"""

import gzip
import json
import pathlib
from typing import Any, Callable

import pandas as pd

from ..logging import logger
from ..main import DataSet, Feature, FeatureInput, Phenotype, PhenotypeInput
from ..training import Predictor, Score
from ..visualization.feature_importances import plot_shap_summary
from .config import Config


class PredictionPipeline:
    def __init__(
        self,
        config_path: pathlib.Path,
        pinputs: list[PhenotypeInput],
        finputs: list[FeatureInput],
        make_classifier: Callable[[int, list[str] | None], Any],
        output_dir: pathlib.Path,
        n_cpus: int,
    ):
        self.config = Config.load_config(config_path)
        self.dataset = DataSet.read_data(pinputs, finputs)
        self.make_classifier = make_classifier
        self.output_dir = output_dir
        self.n_cpus = n_cpus

    @staticmethod
    def is_ydata_good(phenotype: Phenotype, config: Config) -> bool:
        """Check if the phenotype data is good for training.

        Parameters
        ----------
        phenotype : Phenotype
            Phenotype object.

        Returns
        -------
        bool
            True if the data is good for training, otherwise False.
        """
        phenotype_data = phenotype.phenotype_data
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
    def preprocess_feature_data(feature: Feature, config: Config):
        """Preprocess the data."""
        feature_data = feature.feature_data
        # Variance filtering
        feature_data, low_var_features = feature.remove_features_with_low_variance(
            feature_data, config.variance_threshold
        )
        # Correlation filtering
        if config.correlation_threshold is not None:
            feature_data, corr_group_dict = (
                feature.remove_features_with_high_correlation(
                    feature_data, config.correlation_threshold
                )
            )

    @staticmethod
    def select_features(feature: Feature, phenotype: Phenotype, config: Config):
        feature_data = feature.feature_data
        phenotype_data = phenotype.phenotype_data
        if config.score_function is not None:
            feature_data, low_score_features = feature.feature_selection_kbest(
                feature_data, phenotype_data, config.score_function
            )
        if config.reduction_function is not None:
            feature_data = feature.feature_dimensionality_reduction(
                feature_data, config.reduction_function, config.n_feature_reduction
            )

    def run(self):
        """Run the pipeline."""
        for feature in self.dataset.feature_set:
            self.preprocess_feature_data(feature, self.config)
            for phenotype in self.dataset.phenotype_set:
                if not self.is_ydata_good(phenotype, self.config):
                    return None
                self.select_features(feature, phenotype, self.config)

    @staticmethod
    def save_preprocessing_data(
        output_dir: pathlib.Path,
        low_var_features: list[str],
        correlated_features_dict: dict[str, list[str]],
        low_score_features: list[str],
    ):
        """Save the preprocessing data.

        Parameters
        ----------
        output_dir : pathlib.Path
            The output directory.
        low_var_features : list[str]
            The features with low variance that were removed.
        correlated_features_dict : dict[str, list[str]]
            The features with high correlation that were removed.
        low_score_features : list[str]
            The features with low score_func score that were removed.
        """
        with open(output_dir / "low_var_features_list.txt", "w") as fid:
            fid.write("\n".join(low_var_features))
        with gzip.open(output_dir / "corr_features_map.json.gz", "wt") as gzfile:
            json.dump(correlated_features_dict, gzfile)
        with open(output_dir / "low_score_features_list.txt", "w") as fid:
            fid.write("\n".join(low_score_features))

    @staticmethod
    def save_data(
        output_dir: pathlib.Path,
        predictor: Predictor,
        score: Score,
        save_estimators: bool = False,
    ):
        """Save the data (after training and scoring).

        Parameters
        ----------
        output_dir : pathlib.Path
            The output directory.
        predictor : Predictor
            The predictor object.
        score : Score
            The score object.
        save_estimators : bool
            Whether to save the estimators.
        """
        # save the training data
        if predictor.training_data is None:
            raise ValueError("Training data not set for the predictor")
        predictor.training_data.save_indices(output_dir)
        # save the cv data
        if predictor.cv_data is None:
            raise ValueError("CV data not set for the predictor")
        predictor.cv_data.save_indices(predictor.phenotype, output_dir)
        # save the scores
        score.save_scores(output_dir)
        # save estimators
        if save_estimators:
            score.save_estimators(output_dir)

    @staticmethod
    def save_visualizations(
        output_dir: pathlib.Path, clf, X_train: pd.DataFrame, config: Config, title: str
    ) -> None:
        """Save the visualizations.

        Parameters
        ----------
        clf
            The classifier object.
        output_dir : pathlib.Path
            The output directory.
        X_train : pd.DataFrame
            The training data.
        config : Config
            The configuration object.
        title : str
            The title of the plot (phenotype name).
        """
        # TODO: Make the file name variable if you want to save the shap plots for each CV
        shap_summary_plot_file = str(output_dir / "shap_summary_plot.png")
        importance_df = plot_shap_summary(
            clf,
            X_train,
            config,
            title=title,
            output_file=shap_summary_plot_file,
        )
        importance_df.to_csv(output_dir / "shap_features.csv", index=True, sep=",")

    # TODO: Create a run method

    # TODO: Remove these and add them directly to the preprocess_data method
    def filter_feature_data(
        self,
        variance_threshold: float | None = 0.05,
        correlation_treshold: float | None = 0.95,
        score_func: str | None = None,
        n_features: int = 1000,
        method: str = "numpy",
    ) -> tuple[list[str], dict[str, list[str]], list[str]]:
        if self._feature_data is not None:
            if variance_threshold is not None:
                fd_high_var, low_var_features = remove_features_with_low_variance(
                    self._feature_data, variance_threshold
                )
            else:
                fd_high_var = self._feature_data
                low_var_features = []
            if correlation_treshold is not None:
                (
                    fd_high_var_low_corr,
                    corr_group_dict,
                ) = remove_features_with_high_correlation(
                    fd_high_var, correlation_treshold, method=method
                )
            else:
                fd_high_var_low_corr = fd_high_var
                corr_group_dict = {}
            if score_func is not None:
                fd_final, low_score_features = feature_selection_kbest(
                    fd_high_var_low_corr, self.phenotype_data, score_func, n_features
                )
            else:
                fd_final = fd_high_var_low_corr
                low_score_features = []
            self._feature_data = fd_final
        else:
            raise ValueError("Feature data not set for this phenotype")
        return low_var_features, corr_group_dict, low_score_features

    def reduce_feature_data(
        self,
        method: str,
        n_components: int,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        """
        Reduces the dimensionality of the feature data for this phenotype.

        Parameters
        ----------
        method : str
            Method to use for dimensionality reduction.
            Supported methods are 'PCA' and 'NMF'
        n_components : int
            Number of components to reduce to.
        random_state : int | None, optional
            Random seed value, by default None

        Returns
        -------
        components_df : pd.DataFrame
            Pandas DataFrame containing the components of the dimensionality reduction.

        Raises
        ------
        ValueError
            If feature data is not set for this phenotype
        """
        if self._feature_data is not None:
            reduced_feature_df, components_df = feature_dimensionality_reduction(
                self._feature_data, method, n_components, random_state=random_state
            )
            self._feature_data = reduced_feature_df
            return components_df
        else:
            raise ValueError("Feature data not set for this phenotype")

    def select_important_features(
        self, feature_importances: np.ndarray, k: int
    ) -> pd.DataFrame:
        """
        Selects the k most important features for this phenotype using feature_importances

        Parameters
        ---------
        feature_importances : np.ndarray
            Numpy array containing the feature importances
        k : int
            Number of features to select

        Returns
        ------
        pd.DataFrame
            Pandas DataFrame containing the selected features
        """
        feature_data = self.feature_data
        importance_df = pd.DataFrame(
            {"feature": feature_data.columns, "importance": feature_importances}
        ).sort_values(by="importance", ascending=False)
        selected_features = importance_df["feature"].tolist()[:k]
        self._feature_data = feature_data[selected_features]
        return self._feature_data
