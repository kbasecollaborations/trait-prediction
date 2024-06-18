"""Module that contains functions for filtering features using correlation threshold"""

from typing import Iterator

import networkx as nx
import numpy as np
import polars as pl
from numba import njit, prange


@njit
def _row_mean(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0],))
    for i in range(a.shape[0]):
        out[i] = np.mean(a[i])
    return out


@njit
def _row_std(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0],))
    for i in range(a.shape[0]):
        out[i] = np.std(a[i])
    return out


@njit
def _tilde(x: np.ndarray, m: int) -> np.ndarray:
    x_mean: np.ndarray = _row_mean(x).reshape(m, 1)
    x_std: np.ndarray = _row_std(x).reshape(m, 1)
    return (x - x_mean) / x_std


@njit
def _pearson_correlation_coefficient(x: np.ndarray) -> np.ndarray:
    m, n = x.shape[0], x.shape[1]  # number of features, number of data points
    x_tilde = _tilde(x, m)
    pcc = x_tilde @ x_tilde.T / n
    np.fill_diagonal(pcc, 1.0)
    return pcc


@njit(parallel=True)
def _row_mean_par(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0],))
    for i in prange(a.shape[0]):
        out[i] = np.mean(a[i])
    return out


@njit(parallel=True)
def _row_std_par(a: np.ndarray) -> np.ndarray:
    out = np.zeros((a.shape[0],))
    for i in prange(a.shape[0]):
        out[i] = np.std(a[i])
    return out


@njit
def _tilde_par(x: np.ndarray, m: int) -> np.ndarray:
    x_mean: np.ndarray = _row_mean_par(x).reshape(m, 1)
    x_std: np.ndarray = _row_std_par(x).reshape(m, 1)
    return (x - x_mean) / x_std


@njit
def _pearson_correlation_coefficient_par(x: np.ndarray) -> np.ndarray:
    m, n = x.shape[0], x.shape[1]  # number of features, number of data points
    x_tilde = _tilde_par(x, m)
    pcc = x_tilde @ x_tilde.T / n
    np.fill_diagonal(pcc, 1.0)
    return pcc


@njit
def _argwhere(pc_mat: np.ndarray, threshold: float) -> np.ndarray:
    return np.argwhere(np.triu(pc_mat, 1) >= threshold)


class GraphCorrelationFilter:
    def __init__(
        self, df: pl.DataFrame, id_col: str, threshold: float, parallel: bool = False
    ) -> None:
        self._df = df
        self._features = [col for col in df.columns if col != id_col]
        self._graph = self._create_graph(df, id_col, threshold, parallel)
        self._ready = True

    @staticmethod
    def _create_graph(
        df: pl.DataFrame, id_col: str, threshold: float, parallel: bool
    ) -> nx.Graph:
        """Create a graph from the given DataFrame using the Pearson correlation coefficient.

        Parameters
        ----------
        df : pl.DataFrame
            The DataFrame containing the data.
        id_col : str
            The column name of the ID column.
        threshold : float
            The threshold for the Pearson correlation coefficient.
        parallel : bool
            Whether to use parallel computation.

        Returns
        -------
        nx.Graph
            The graph.
        """
        if parallel:
            pcc_func = _pearson_correlation_coefficient_par
        else:
            pcc_func = _pearson_correlation_coefficient
        pc_mat = pcc_func(df.drop(id_col).transpose().to_numpy())
        assert pc_mat.shape[0] + 1 == len(df.columns)
        assert pc_mat.shape[1] + 1 == len(df.columns)
        graph = nx.Graph()
        for c in _argwhere(pc_mat, threshold):
            graph.add_edge(c[0], c[1])
        return graph

    def _check_ready(self) -> None:
        """Check if the graph is ready to be accessed."""
        if not self._ready:
            raise ValueError("Graph has been consumed")

    @property
    def graph(self) -> nx.Graph:
        """The graph."""
        self._check_ready()
        return self._graph.copy()

    def find_corr_cols(self) -> dict[str, list[str]]:
        """Find the correlated columns.

        Returns
        -------
        dict[str, list[str]]
            The dictionary that maps the column names to the correlated columns.
        """
        self._check_ready()
        G = self.graph
        neighbor_col_dict = {
            self._features[node]: [
                self._features[neighbor] for neighbor in list(G.neighbors(node))
            ]
            for node in G.nodes()
        }
        return neighbor_col_dict

    def _redundant_nodes_iter(self) -> Iterator[int]:
        """Iterate over the redundant nodes."""
        degrees = sorted(
            self._graph.degree(self._graph.nodes), key=lambda v: v[1], reverse=True
        )
        # Do not track 0-degree nodes
        try:
            degree_list: list[list[int]] = [[] for _ in range(degrees[0][1])]  # type: ignore
        except IndexError:
            return []
        for node_name, deg in degrees:
            if deg == 0:
                break
            degree_list[deg - 1].append(node_name)  # type: ignore
        current_degree_idx = len(degree_list) - 1
        while current_degree_idx >= 0:
            if len(degree_list[current_degree_idx]) == 0:
                current_degree_idx -= 1
                continue
            current_degree_list = degree_list[current_degree_idx]
            while len(current_degree_list) > 0:
                entry = current_degree_list.pop()
                if self._graph.degree[entry] == 0:
                    continue
                is_current = self._graph.degree[entry] == current_degree_idx + 1
                if is_current:
                    self._graph.remove_node(entry)
                    yield entry
                else:
                    updated_degree = self._graph.degree[entry]
                    degree_list[updated_degree - 1].append(entry)  # type: ignore
        self._ready = False

    def remove_corr_cols(self) -> pl.DataFrame:
        """Remove the correlated columns from the DataFrame.

        Returns
        -------
        pl.DataFrame
            The DataFrame with the correlated columns removed.
        """
        self._check_ready()
        to_drop = [self._features[node] for node in self._redundant_nodes_iter()]
        return self._df.drop(to_drop)
