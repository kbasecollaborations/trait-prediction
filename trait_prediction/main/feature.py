"""Module that defines the Feature class"""

import pathlib
from dataclasses import dataclass
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

from .feature_preprocessing import (
    _find_corr_cols_numba,
    _find_corr_cols_numba_parallel,
    _find_corr_cols_numpy,
    _pearson_correlation_coefficient,
)


@dataclass(frozen=True)
class FeatureIndex:
    """Class that represents a feature index.

    Attributes
    ----------
    name : str
        The name of the feature.
    ftype : str
        The type of the feature data. Either 'count', 'float' or 'binary'.
    dtype : str
        The data type of the feature data.
    """

    name: str
    ftype: str
    dtype: str


@dataclass(frozen=True)
class FeatureInput:
    """Class that represents a feature input.

    Attributes
    ----------
    path : pathlib.Path | str
        The path to the feature data.
    findex : FeatureIndex
        The FeatureIndex object containing the name, ftype and dtype of the feature.
    index_format_func : Callable[[str], str]
        Function to format the index of the feature data.
        Eg: lambda x: x.strip().split("?")[-1].removesuffix(".RAST").removesuffix(".fna")
    """

    path: pathlib.Path | str
    findex: FeatureIndex
    index_format_func: Callable[[str], str]


class Feature:
    """The Feature class.

    Parameters
    ----------
    raw_feature_data : pd.DataFrame
        The raw feature data.
    findex : FeatureIndex
        The FeatureIndex object containing the name, ftype and dtype of the feature.

    Attributes
    ----------
    findex : FeatureIndex
        The FeatureIndex object containing the name, ftype and dtype of the feature.
    feature_data : pd.DataFrame
        Pandas DataFrame containing the feature data.
    """

    def __init__(self, raw_feature_data: pd.DataFrame, findex: FeatureIndex) -> None:
        self.findex = findex
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
        undup_raw_feature_data = raw_feature_data.dropna().astype(self.findex.dtype)
        return undup_raw_feature_data.loc[
            ~undup_raw_feature_data.index.duplicated(keep="first"), :
        ]

    def __repr__(self) -> str:
        n_genomes = self._feature_data.shape[0]
        n_features = self._feature_data.shape[1]
        return (
            f"Feature (name={self.findex.name}, type={self.findex.ftype},"
            f" dtype={self.findex.dtype}, n_genomes={n_genomes}, n_features={n_features})"
        )

    def __hash__(self) -> int:
        return hash(self.findex)

    @classmethod
    def read_data(
        cls,
        finput: FeatureInput,
    ) -> "Feature":
        """Reads feature data from a TSV file and returns a Feature object.

        Parameters
        ----------
        finput : FeatureInput
            Feature input containing the path, FeatureIndex and index_format_func.

        Returns
        -------
        Feature
            The Feature object.
        """
        file_path = finput.path
        ftype = finput.findex.ftype
        index_format_func = finput.index_format_func
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
        feature_df = feature_df.set_index(id_col).fillna(0).astype(final_dtype)
        if ftype == "binary":
            feature_df[feature_df > 0] = 1
        feature_df.index.name = "genomeID"
        return cls(feature_df, finput.findex)

    @classmethod
    def merge_data(cls, finputs: tuple[FeatureInput, ...]) -> "Feature":
        """Merges feature data (concat rows) from multiple FeatureInput objects and returns a Feature object.

        Parameters
        ----------
        finputs : tuple[FeatureInput, ...]
            Tuple of FeatureInput objects.

        Returns
        -------
        Feature
            The Feature object.
        """
        features: list[Feature] = []
        for finput in finputs:
            feature = cls.read_data(finput)
            features.append(feature)
        feature_data = pd.concat([feature.feature_data for feature in features]).fillna(
            0
        )
        if len(set([feature.findex.name for feature in features])) > 1:
            raise ValueError("The features must have the same name")
        findex = FeatureIndex(
            name=features[0].findex.name,
            ftype=features[0].findex.ftype,
            dtype=features[0].findex.dtype,
        )
        return cls(feature_data, findex)

    @property
    def feature_data(self) -> pd.DataFrame:
        """Pandas DataFrame containing the feature data."""
        return self._feature_data.copy(deep=True)

    @staticmethod
    def remove_features_with_low_variance(
        feature_df: pd.DataFrame, threshold: float = 0.05
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Removes features with low variance from the given feature DataFrame.

        Parameters
        ---------
        feature_df : pd.DataFrame
            Pandas DataFrame containing the feature data.
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
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(feature_df)
        mask = vt.get_support()
        if mask is None:
            raise ValueError("No features were selected")
        removed_features = list(feature_df.columns[~mask])
        return feature_df.loc[:, list(mask)], removed_features

    @staticmethod
    def remove_features_with_high_correlation(
        feature_df: pd.DataFrame, threshold: float = 0.95, method: str = "numpy"
    ) -> tuple[pd.DataFrame, dict[str, list[str]]]:
        """
        Removes features with high correlation from the given feature DataFrame.

        Parameters
        ---------
        feature_df : pd.DataFrame
            Pandas DataFrame containing the feature data.
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

    @staticmethod
    def feature_selection_kbest(
        feature_df: pd.DataFrame,
        target_s: pd.Series,
        score_func: str,
        n_features: int = 1000,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Select features according to the k highest score_func scores.

        Parameters
        ---------
        feature_df : pd.DataFrame
            Pandas DataFrame containing the feature data.
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
        support = kbest.get_support()
        if support is None:
            raise ValueError("No features were selected")
        removed_features = list(feature_df.columns[~support])
        return kbest_feature_df, removed_features

    @staticmethod
    def feature_dimensionality_reduction(
        feature_df: pd.DataFrame,
        method: str,
        n_components: int,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform dimensionality reduction on the given feature DataFrame.

        Parameters
        ---------
        feature_df : pd.DataFrame
            Pandas DataFrame containing the feature data.
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
