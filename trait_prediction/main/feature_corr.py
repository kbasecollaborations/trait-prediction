"""Module that contains functions for preprocessing features"""

import numpy as np
from numba import jit, prange, typed, types


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
