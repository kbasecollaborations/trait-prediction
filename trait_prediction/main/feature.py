"""Module that defines the Feature class"""

import pathlib
from typing import Callable
from warnings import warn

import pandas as pd
import polars as pl
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)

from .feature_selection import (
    _find_corr_cols_numba,
    _find_corr_cols_numba_parallel,
    _find_corr_cols_numpy,
    _pearson_correlation_coefficient,
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
        if ftype == "binary":
            feature_df[feature_df > 0] = 1
        feature_df = feature_df.set_index(header[0]).fillna(0).astype(final_dtype)
        feature_df.index.name = "genomeID"
        return cls(feature_df, name, ftype, dtype)

    @property
    def feature_data(self) -> pd.DataFrame:
        """Pandas DataFrame containing the feature data."""
        return self._feature_data.copy(deep=True)

    def remove_features_with_low_variance(
        self, threshold: float = 0.05
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Removes features with low variance from the given feature DataFrame.

        Parameters
        ---------
        threshold : float
            Threshold for the variance of the features.
            Default value is 0.05 which removes features with more than 95% of the values being the same.

        Returns
        ------
        pd.DataFrame
            Pandas DataFrame containing the features with low variance removed.
        list[str]
            List of the features with low variance that were removed
        """
        feature_df = self.feature_data
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(feature_df)
        mask = vt.get_support()
        if mask is None:
            raise ValueError("No features were selected")
        removed_features = list(feature_df.columns[~mask])
        return feature_df.loc[:, list(mask)], removed_features

    def remove_features_with_high_correlation(
        self, threshold: float = 0.95, method: str = "numpy"
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Removes features with high correlation from the given feature DataFrame.

        Parameters
        ---------
        threshold : float
            Threshold for the correlation of the features.
            Default value is 0.95
        method : str, optional
            Method used to calculate correlated features
            Options available are 'numpy', 'numba_parallel' and 'numba'
            Default value is 'numpy'

        Returns
        ------
        pd.DataFrame
            Pandas DataFrame containing the features with high correlation removed.
        dict[str, list[str]]
            Dictionary of the features with high correlation that were removed
        """
        feature_df = self.feature_data
        # Get correlated columns
        if method == "numba_parallel":
            corr_cols = _find_corr_cols_numba_parallel(feature_df.to_numpy(), threshold)
        elif method == "numba":
            corr_cols = _find_corr_cols_numba(feature_df.to_numpy(), threshold)
        elif method == "numpy":
            corr_matrix = _pearson_correlation_coefficient(feature_df.to_numpy().T)
            corr_cols = _find_corr_cols_numpy(corr_matrix, threshold)
        else:
            raise ValueError(
                f"method {method} is not supported. Supported methods are numpy, numba_parallel, numba"
            )
        # Find columns to drop
        cols_to_drop = set()
        corr_group_dict = dict()
        cols = feature_df.columns
        for i, corr_col in enumerate(corr_cols):
            if len(corr_col) < 1:
                continue
            cols_to_drop.update(corr_col)
            corr_group_dict[cols[i]] = list(cols[corr_col])
        # Drop the columns from the DataFrame
        return (
            feature_df.drop(list(feature_df.columns[list(cols_to_drop)]), axis=1),
            corr_group_dict,
        )

    def feature_selection_kbest(
        self,
        target_s: pd.Series,
        score_func: str,
        n_features: int = 1000,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select features according to the k highest score_func scores.

        Parameters
        ---------
        target_s : pd.Series
            Pandas Series containing the target.
        score_func : str
            Supported values are 'f_classif', 'mutual_info_classif', 'chi2'
            Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores.
        n_features : int, optional
            Number of features to select.
            Default value is 1000.

        Returns
        ------
        pd.DataFrame
            Pandas DataFrame containing the features with high score_func scores.
        list[str]
            List of removed features with low score_func score
        """
        feature_df = self.feature_data
        if score_func == "f_classif":
            score_function = f_classif
        elif score_func == "mutual_info_classif":
            score_function = mutual_info_classif
        elif score_func == "chi2":
            score_function = chi2
        else:
            raise ValueError(
                f"score_func {score_func} is not supported. Supported functions are f_classif, mutual_info_classif, chi2"
            )
        kbest = SelectKBest(score_func=score_function, k=n_features)
        kbest_features = kbest.fit_transform(feature_df, target_s)
        kbest_feature_df = pd.DataFrame(
            kbest_features,
            columns=feature_df.columns[kbest.get_support()],
            index=feature_df.index,
        )
        if kbest.get_support() is None:
            raise ValueError("No features were selected")
        removed_features = list(feature_df.columns[~kbest.get_support()])
        return kbest_feature_df, removed_features

    def feature_dimensionality_reduction(
        self,
        method: str,
        n_components: int,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform dimensionality reduction on the given feature DataFrame.

        Parameters
        ---------
        method : str
            Supported values are 'NMF', 'PCA'
        n_components : int
            Number of components to reduce to.
        random_state : int | None
            Seed for the random number generator.
            Default value is None.

        Returns
        ------
        reduced_feature_df : pd.DataFrame
            Pandas DataFrame containing the features with reduced dimensionality.
        components_df : pd.DataFrame
            Pandas DataFrame containing the components of the dimensionality reduction.
        """
        feature_df = self.feature_data
        min_dim = min(feature_df.shape)
        if n_components > min_dim:
            warn(
                f"n_components {n_components} is greater than the maximum dimension {min_dim} using {min_dim} instead"
            )
            n_comps = min_dim
        else:
            n_comps = n_components
        if method == "NMF":
            model = NMF(n_components=n_comps, init="nndsvd", random_state=random_state)  # type: ignore
            prefix = "NMF"
        elif method == "PCA":
            model = PCA(n_components=n_comps, random_state=random_state)
            prefix = "PCA"
        else:
            raise ValueError(
                f"method {method} is not supported. Supported methods are NMF, PCA"
            )
        reduced_features = model.fit_transform(feature_df)
        reduced_feature_df = pd.DataFrame(
            reduced_features,
            columns=pd.Index(
                [f"{prefix}_{i}" for i in range(reduced_features.shape[1])]
            ),
            index=feature_df.index,
        )
        components = model.components_
        components_df = pd.DataFrame(
            components,
            columns=feature_df.columns,
            index=pd.Index([f"{prefix}_{i}" for i in range(components.shape[0])]),
        )
        return reduced_feature_df, components_df
