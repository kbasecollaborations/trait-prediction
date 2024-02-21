"""Module that contains functions for eliminating features"""

from warnings import warn

import numpy as np
import pandas as pd
from numba import jit, prange, typed, types
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


@jit(nopython=True, parallel=True)
def _find_corr_cols_numba_parallel(
    matrix: np.ndarray, threshold=0.95
) -> list[list[int]]:
    """
    Find columns to drop based on correlation threshold (parallelism enabled)

    Parameters
    ----------
    matrix : np.ndarray
        2-dimensional matrix. M (data points) x N (features)
    threshold : float, optional
        Correlation threshold, by default 0.95

    Returns
    -------
    corr_cols
        List of lists containing correlated columns for each columns
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    corr_cols = [typed.List.empty_list(types.int64) for _ in range(n)]
    for i in prange(n):
        to_drop = typed.List.empty_list(types.int64)
        for j in range(i + 1, n):
            xi, xj = matrix[:, i], matrix[:, j]
            mean_i, mean_j = np.mean(xi), np.mean(xj)
            std_i, std_j = np.std(xi), np.std(xj)
            corr = np.sum((xi - mean_i) * (xj - mean_j)) / (m * std_i * std_j)  # type: ignore
            if corr >= threshold:
                to_drop.append(j)
        corr_cols[i] = to_drop
    return corr_cols


@jit(nopython=True)
def _find_corr_cols_numba(matrix: np.ndarray, threshold=0.95) -> list[list[int]]:
    """
    Find columns to drop based on correlation threshold (parallelism disabled)

    Parameters
    ----------
    matrix : np.ndarray
        2-dimensional matrix. M (data points) x N (features)
    threshold : float, optional
        Correlation threshold, by default 0.95

    Returns
    -------
    corr_cols
        List of lists containing correlated columns for each columns
    """
    m = matrix.shape[0]
    n = matrix.shape[1]
    corr_cols = [typed.List.empty_list(types.int64) for _ in range(n)]
    for i in range(n):
        to_drop = typed.List.empty_list(types.int64)
        if i >= 1:
            dropped_so_far_list = typed.List.empty_list(types.int64)
            for corr_list in corr_cols:
                for elem in corr_list:
                    dropped_so_far_list.append(elem)
            if i in set(dropped_so_far_list):
                continue
        for j in range(i + 1, n):
            xi, xj = matrix[:, i], matrix[:, j]
            mean_i, mean_j = np.mean(xi), np.mean(xj)
            std_i, std_j = np.std(xi), np.std(xj)
            corr = np.sum((xi - mean_i) * (xj - mean_j)) / (m * std_i * std_j)  # type: ignore
            if corr >= threshold:
                to_drop.append(j)
        corr_cols[i] = to_drop
    return corr_cols


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
def _find_corr_cols_numpy(corr_matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """
    Find columns to drop based on correlation matrix and threshold.

    Parameters
    ----------
    corr_matrix : numpy.ndarray
        The M rows and M columns (number of features).
    threshold : float
        The correlation threshold.

    Returns
    -------
    corr_cols
        List of lists containing correlated columns for each columns
    """
    n_cols = corr_matrix.shape[1]
    corr_cols = [typed.List.empty_list(types.int64) for _ in range(n_cols)]
    for i in range(n_cols):
        to_drop = typed.List.empty_list(types.int64)
        if i >= 1:
            dropped_so_far_list = typed.List.empty_list(types.int64)
            for corr_list in corr_cols:
                for elem in corr_list:
                    dropped_so_far_list.append(elem)
            if i in set(dropped_so_far_list):
                continue
        for j in range(i + 1, n_cols):
            if corr_matrix[i, j] >= threshold:
                to_drop.append(j)
        corr_cols[i] = to_drop
    return corr_cols


# TODO: FIXME: The corr_dict and cols_to_drop are not accurate
# We need to ignore the cols added to cols_to_drop while running the loop
# Also we need to run the inner loop over all the columns in corr_matrix
def remove_features_with_high_correlation(
    feature_df: pd.DataFrame, threshold: float = 0.95, method: str = "numpy"
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
        feature_df.drop(feature_df.columns[list(cols_to_drop)], axis=1),
        corr_group_dict,
    )


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
    min_dim = min(feature_df.shape)
    if n_components > min_dim:
        warn(
            f"n_components {n_components} is greater than the maximum dimension {min_dim} using {min_dim} instead"
        )
        n_comps = min_dim
    else:
        n_comps = n_components
    if method == "NMF":
        model = NMF(n_components=n_comps, init="nndsvd", random_state=random_state)
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
