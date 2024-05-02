"""Module that defines the Feature class"""

import pathlib
from typing import Callable

import numpy as np
import pandas as pd
import polars as pl

# TODO: Handle these functions within the class
from ..feature_selection.reduction import (
    feature_dimensionality_reduction,
    feature_selection_kbest,
    remove_features_with_high_correlation,
    remove_features_with_low_variance,
)


class Feature:
    """The Feature class.

    Parameters
    ----------
    raw_feature_data : pd.DataFrame
        The raw feature data.
    name : str
        The name of the feature.
    ftype : str
        The type of the feature data. Either 'count', 'float' or 'binary'.
    dtype : str
        The data type of the feature data.

    Attributes
    ----------
    name : The name of the feature.
    ftype : The type of the feature data. Either 'count', 'float' or 'binary'.
    dtype : The data type of the feature data.
    """

    def __init__(
        self, raw_feature_data: pd.DataFrame, name: str, ftype: str, dtype: str
    ) -> None:
        self.name = name
        self.ftype = ftype
        self.dtype = dtype
        self._feature_data: pd.DataFrame = self._parse_feature_data(raw_feature_data)

    def _parse_feature_data(self, raw_feature_data: pd.DataFrame) -> pd.DataFrame:
        """Parses the given raw feature data.

        Parameters
        ----------
        raw_feature_data : pd.DataFrame
            Pandas DataFrame containing the raw feature data.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing the filtered feature data.
        """
        undup_raw_feature_data = raw_feature_data.dropna().astype(self.dtype)
        return undup_raw_feature_data.loc[
            ~undup_raw_feature_data.index.duplicated(keep="first"), :
        ]

    def __repr__(self) -> str:
        n_genomes = self._feature_data.shape[0]
        n_features = self._feature_data.shape[1]
        return f"Feature (name={self.name}, type={self.ftype}, dtype={self.dtype}, n_genomes={n_genomes}, n_features={n_features})"

    def __hash__(self) -> int:
        unique_id = {
            "name": self.name,
            "ftype": self.ftype,
            "dtype": self.dtype,
        }
        return hash(unique_id)

    @classmethod
    def read_data(
        cls,
        file_path: str | pathlib.Path,
        name: str,
        ftype: str,
        dtype: str,
        index_format_func: Callable[[str], str],
    ) -> "Feature":
        """Reads feature data from a TSV file and returns a Feature object.

        Parameters
        ----------
        file_path : str | pathlib.Path
            Path to the TSV file containing the feature data.
        name : str
            Name of the feature.
        ftype : str
            Type of the feature data.
            Either 'count', 'float' or 'binary'.
        dtype : str
            Data type of the feature data.
        index_format_func : Callable[[str], str]
            Function to format the index of the feature data.
            Eg: lambda x: x.strip().split("?")[-1].removesuffix(".RAST").removesuffix(".fna")

        Returns
        -------
        Feature
            The Feature object.
        """
        with open(file_path, "r") as fid:
            header = fid.readline().strip().split("\t")
        dtypes = dict()
        id_col = header[0]
        dtypes[header[0]] = pl.String
        if ftype == "float":
            final_dtype = "float64"
        elif ftype == "binary":
            final_dtype = "uint8"
        elif ftype == "count":
            final_dtype = "uint32"
        else:
            raise ValueError(
                f"Invalid feature type: {ftype}, must be 'count', 'float' or 'binary'"
            )
        # NOTE: We use Float64 while reading the data so that it is faster
        for col in header[1:]:
            dtypes[col] = pl.Float64
        feature_df = pl.read_csv(
            file_path,
            has_header=True,
            separator="\t",
            columns=header,
            dtypes=dtypes,
            use_pyarrow=False,
        ).to_pandas()
        feature_df[id_col] = feature_df[id_col].apply(index_format_func)
        feature_df = feature_df.set_index(header[0]).fillna(0).astype(final_dtype)
        feature_df.index.name = "genomeID"
        return cls(feature_df, name, ftype, dtype)

    @property
    def feature_data(self) -> pd.DataFrame:
        """Pandas DataFrame containing the feature data."""
        return self._feature_data.copy(deep=True)

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
