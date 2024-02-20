"""Module that defines the Phenotype class"""

import pathlib
import pickle

import numpy as np
import pandas as pd

from ..feature_selection.reduction import (
    feature_dimensionality_reduction,
    feature_selection_kbest,
    remove_features_with_high_correlation,
    remove_features_with_low_variance,
)

# TODO: Write test. use copilot to write test


class Phenotype:
    """
    Class that represents a phenotype.

    Parameters
    ---------
    raw_phenotype_data : pd.Series
        Pandas Series containing the raw phenotype data.
    name : str
        Name of the phenotype.
    category : str
        Category of the phenotype.

    Attributes
    ---------
    phenotype_data : pd.Series
        Pandas Series containing the filtered phenotype data.
    name : str
        Name of the phenotype.
    category : str
        Category of the phenotype.
    feature_data : pd.DataFrame
        Pandas DataFrame containing the feature data.
    feature_type : str | None
        Type of the feature data.
    """

    def __init__(self, raw_phenotype_data: pd.Series, name: str, category: str) -> None:
        self.name = name
        self.category = category
        self._phenotype_data = self._parse_phenotype_data(raw_phenotype_data)
        self._feature_data: None | pd.DataFrame = None
        self.feature_type: None | str = None
        self._common_genomes: None | list[str] = None

    def __repr__(self) -> str:
        if self._common_genomes is None:
            size = self._phenotype_data.shape[0]
        else:
            size = len(self._common_genomes)
        return f"Phenotype (name={self.name}, category={self.category}, size={size})"

    def __hash__(self) -> int:
        unique_id = {
            "name": self.name,
            "category": self.category,
            "feature_type": self.feature_type,
        }
        return hash(unique_id)

    @property
    def phenotype_data(self) -> pd.Series:
        """Pandas Series containing the filtered phenotype data."""
        if self._common_genomes is None:
            return self._phenotype_data
        else:
            return self._phenotype_data.loc[self._common_genomes].dropna().copy()

    def _parse_phenotype_data(self, raw_phenotype_data: pd.Series) -> pd.Series:
        """
        Parses the given raw phenotype data.

        Parameters
        ---------
        raw_phenotype_data : pd.Series
            Pandas Series containing the raw phenotype data.

        Returns
        ------
        pd.Series
            Pandas Series containing the filtered phenotype data.
        """
        undup_raw_phenotype_data = raw_phenotype_data.dropna().astype("uint8")
        return undup_raw_phenotype_data[
            ~undup_raw_phenotype_data.index.duplicated(keep="first")
        ]

    def set_feature_data(
        self, raw_feature_data: pd.DataFrame, feature_type: str, force: bool = False
    ) -> None:
        """
        Sets the feature data for this phenotype.

        Parameters
        ---------
        raw_feature_data : pd.DataFrame
            Pandas DataFrame containing the feature data.
        feature_type : str
            Type of the feature data.
        force : bool
            If True, the feature data will be set even if it was already set.
            Default value is False

        Note
        ----
        This method will overwrite self.phenotype_data and self.feature_data (force=True).
        """
        if self._feature_data is None or force:
            undup_raw_feature_data = raw_feature_data[
                ~raw_feature_data.index.duplicated(keep="first")
            ]
            common_genomes = sorted(
                list(
                    set(self.phenotype_data.index).intersection(
                        set(undup_raw_feature_data.index)
                    )
                )
            )
            self._common_genomes = common_genomes
            self._feature_data = undup_raw_feature_data.loc[common_genomes, :]
            self.feature_type = feature_type
        else:
            raise ValueError("Feature data already set for this phenotype")

    @property
    def feature_data(self) -> pd.DataFrame:
        """Pandas DataFrame containing the feature data."""
        if self._feature_data is None:
            raise ValueError("Feature data not set for this phenotype")
        else:
            return self._feature_data.dropna().copy()

    def unset_feature_data(self) -> None:
        """
        Unsets the feature data for this phenotype.
        """
        self._feature_data = None
        self.feature_type = None
        self._common_genomes = None

    def filter_feature_data(
        self,
        variance_threshold: float | None = 0.05,
        correlation_treshold: float | None = 0.95,
        score_func: str | None = None,
        n_features: int = 1000,
        method: str = "numpy",
    ) -> tuple[list[str], dict[str, list[str]], list[str]]:
        """
        Filters the feature data for this phenotype.

        Parameters
        ---------
        variance_threshold : float | None
            Threshold for the variance of the features.
            Default value 0.05
        correlation_treshold : float | None
            Threshold for the correlation of the features.
            Default value 0.95
        score_func : str, optional
            Supported values are 'f_classif', 'mutual_info_classif', 'chi2'
            Default  value is None
            Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
        n_features : int, optional
            Number of features to select.
            Default value is 1000.
        method : str, optional
            Method used to calculate correlated features
            Options available are 'numpy', 'numba_parallel' and 'numba'
            Default value is 'numpy'

        Returns
        -------
        list[str]
            List of the features with low variance that were removed
        dict[str, list[str]]
            Dictionary of the features with high correlation that were removed
        list[str]
            List of the features with low score_func score that were removed

        """
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

    def save(self, file_path: str | pathlib.Path) -> None:
        """
        Saves the phenotype data to the given path.

        Parameters
        ---------
        file_path : str | pathlib.Path
            The file path to the pickle file along with the extension
        """
        data = {
            "name": self.name,
            "category": self.category,
            "_phenotype_data": self._phenotype_data,
            "_feature_data": self._feature_data,
            "feature_type": self.feature_type,
            "_common_genomes": self._common_genomes,
        }
        with open(file_path, "wb") as fid:
            pickle.dump(data, fid)

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "Phenotype":
        """
        Loads the phenotype data from the given path.

        Parameters
        ---------
        file_path : str | pathlib.Path
            The file path to the pickle file along with the extension

        Returns
        ------
        Phenotype
            Phenotype object
        """
        with open(file_path, "rb") as fid:
            data = pickle.load(fid)
        phenotype = cls(data["_phenotype_data"], data["name"], data["category"])
        if data["_feature_data"] is not None:
            phenotype.set_feature_data(
                data["_feature_data"], data["feature_type"], force=True
            )
            if phenotype._common_genomes != data["_common_genomes"]:
                raise ValueError("Common genomes do not match")
        return phenotype
