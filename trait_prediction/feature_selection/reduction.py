"""Module that contains functions for eliminating features"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif


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
    var_filter = feature_df.var() > threshold  # type: ignore
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
        Default value is 0.95

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
        if len(correlated_cols) > 0:
            corr_group_dict[col] = correlated_cols
            cols_to_drop_set.add(col)
    return feature_df.drop(list(cols_to_drop_set), axis=1), corr_group_dict


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
