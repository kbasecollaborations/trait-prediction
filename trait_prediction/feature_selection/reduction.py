"""Module that contains functions for eliminating features"""

import numpy as np
import pandas as pd
from numba import jit
from sklearn.decomposition import NMF, PCA
from sklearn.feature_selection import (
    SelectKBest,
    VarianceThreshold,
    chi2,
    f_classif,
    mutual_info_classif,
)


def remove_features_with_low_variance(
    feature_df: pd.DataFrame, threshold: float = 0.05
) -> tuple[pd.DataFrame, list[str]]:
    """
    Removes features with low variance from the given feature DataFrame.

    Parameters
    ---------
    feature_df : pd.DataFrame
        Pandas DataFrame containing the features.
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
    removed_features = list(feature_df.columns[~mask])
    return feature_df.loc[:, list(mask)], removed_features


def _pearson_correlation_coefficient(X: np.ndarray) -> np.ndarray:
    """
    Calculate the Pearson correlation coefficient of matrix X.

    Parameters
    ----------
    X : numpy array (MxN)
        The input matrix with M rows (number of features) and N columns (number of data points).

    Returns
    -------
    pcc : numpy array (MxM)
        The Pearson correlation coefficient matrix, where each element (i, j) represents the correlation
        between feature i and feature j in the input matrix X.
    """
    M, N = X.shape[0], X.shape[1]  # number of features, number of data points
    X_mean = np.mean(X, axis=1).reshape(M, 1)
    X_std = np.std(X, axis=1).reshape(M, 1)
    X_tilde = (X - X_mean) / X_std
    pcc = X_tilde @ X_tilde.T / N
    np.fill_diagonal(pcc, 1, wrap=False)
    return pcc


@jit(nopython=True)
def _find_columns_to_drop(
    corr_matrix: np.ndarray, threshold: float
) -> tuple[list, list]:
    """
    Find columns to drop based on correlation matrix and threshold.

    Parameters
    ----------
    corr_matrix : numpy.ndarray
        The correlation matrix.
    threshold : float
        The correlation threshold.

    Returns
    -------
    tuple[list, list]
        A tuple containing a list of columns to drop and a list of correlated column groups.
    """
    # Initialize an empty set to hold indices of columns to drop
    cols_to_drop = set()
    corr_groups = []
    # Get the number of columns in the correlation matrix
    n_cols = corr_matrix.shape[1]
    # Iterate over columns
    for i in range(n_cols):
        corr_set = set()
        for j in range(i + 1, n_cols):
            # If correlation is above the threshold and not already in the list
            if corr_matrix[i, j] >= threshold:
                corr_set.add(j)
                if j not in cols_to_drop:
                    cols_to_drop.add(j)
        corr_groups.append(corr_set)
    return list(cols_to_drop), corr_groups


def _create_corr_group_dict(corr_groups: list, cols: pd.Index) -> dict:
    """
    Convert list of sets of indices to dict of column names

    Parameters
    ----------
    corr_groups : list
        List of sets representing correlated groups.
    cols : pd.Index
        List of column names.

    Returns
    -------
    dict
        Dictionary mapping column names to lists of correlated columns.
    """
    corr_group_dict = {}
    for i, corr_group in enumerate(corr_groups):
        if len(corr_group) < 1:
            continue
        corr_group_dict[cols[i]] = list(cols[list(corr_group)])
    return corr_group_dict


def remove_features_with_high_correlation(
    feature_df: pd.DataFrame, threshold: float = 0.95
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """
    Removes features with high correlation from the given feature DataFrame.

    Parameters
    ---------
    feature_df : pd.DataFrame
        Pandas DataFrame containing the features.
    threshold : float
        Threshold for the correlation of the features.
        Default value is 0.95

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the features with high correlation removed.
    dict[str, list[str]]
        Dictionary of the features with high correlation that were removed
    """
    # Calculate the correlation matrix and convert it to a NumPy array
    corr_matrix = np.abs(_pearson_correlation_coefficient(feature_df.to_numpy().T))
    # Find indices of columns to drop
    to_drop, corr_groups = _find_columns_to_drop(corr_matrix, threshold)
    corr_group_dict = _create_corr_group_dict(corr_groups, feature_df.columns)
    # Drop the columns from the DataFrame
    return feature_df.drop(feature_df.columns[to_drop], axis=1), corr_group_dict


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
        Pandas DataFrame containing the features.
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
    removed_features = list(feature_df.columns[~kbest.get_support()])
    return kbest_feature_df, removed_features


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
        Pandas DataFrame containing the features.
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
    if method == "NMF":
        max_dim = min(feature_df.shape)
        if n_components > max_dim:
            raise ValueError(
                f"n_components {n_components} is greater than the maximum dimension {max_dim}"
            )
        model = NMF(n_components=n_components, init="nndsvd", random_state=random_state)
        prefix = "NMF"
    elif method == "PCA":
        model = PCA(n_components=n_components, random_state=random_state)
        prefix = "PCA"
    else:
        raise ValueError(
            f"method {method} is not supported. Supported methods are NMF, PCA"
        )
    reduced_features = model.fit_transform(feature_df)
    reduced_feature_df = pd.DataFrame(
        reduced_features,
        columns=[f"{prefix}_{i}" for i in range(reduced_features.shape[1])],
        index=feature_df.index,
    )
    components = model.components_
    components_df = pd.DataFrame(
        components,
        columns=feature_df.columns,
        index=[f"{prefix}_{i}" for i in range(components.shape[0])],
    )
    return reduced_feature_df, components_df
