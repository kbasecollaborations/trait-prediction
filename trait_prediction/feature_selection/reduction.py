"""Module that contains functions for eliminating features"""

import numpy as np
import pandas as pd


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

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the features with low variance removed.
    list[str]
        List of the features with low variance that were removed
    """
    var_filter = feature_df.var() > threshold
    removed_features = list(feature_df.columns[~var_filter])
    return feature_df.loc[:, var_filter], removed_features


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

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the features with high correlation removed.
    dict[str, list[str]]
        Dictionary of the features with high correlation that were removed
    """
    corr_matrix = feature_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_group_dict = {}
    cols_to_drop_set = set()
    for col in upper.columns:
        corr_filter = upper[col] > threshold
        correlated_cols = list(upper.columns[corr_filter])
        corr_group_dict[col] = correlated_cols
        cols_to_drop_set.add(col)
    return feature_df.drop(list(cols_to_drop_set), axis=1), corr_group_dict
