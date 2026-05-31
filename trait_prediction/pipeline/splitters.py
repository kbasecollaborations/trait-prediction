"""Data splitting strategies for train/test splits.

This module provides various strategies for splitting data into training and test sets,
including random splits and phylogenetically-aware splits (out-of-clade and in-clade).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import reduce
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

try:
    from ete3 import Tree
except ImportError:
    Tree = Any  # type: ignore


class DataSplitter(ABC):
    """Abstract base class for data splitters.

    Child classes must implement the `split` method. The `generate_splits` method
    can be overridden for efficiency in specific implementations.
    """

    @abstractmethod
    def split(self, samples: Sequence[str], **kwargs: Any) -> NDArray[np.str_]:
        """Generate a single train/test split.

        Parameters
        ----------
        samples
            List of sample identifiers to split.
        **kwargs
            Additional keyword arguments for specific splitter implementations.

        Returns
        -------
        NDArray[np.str_]
            Array of sample identifiers for the test set.
        """
        pass

    def generate_splits(
        self, samples: Sequence[str], n: int, **kwargs: Any
    ) -> list[NDArray[np.str_]]:
        """Generate multiple train/test splits.

        Parameters
        ----------
        samples
            List of sample identifiers to split.
        n
            Number of splits to generate.
        **kwargs
            Additional keyword arguments passed to `split`.

        Returns
        -------
        list[NDArray[np.str_]]
            List of test set sample arrays, one per split.
        """
        return [self.split(samples, **kwargs) for _ in range(n)]


class RandomSplitter(DataSplitter):
    """Random data splitter.

    Randomly samples a fraction of the data for the test set.

    Parameters
    ----------
    test_set_ratio
        Fraction of samples to include in the test set (0.0 to 1.0).
    random_state
        Random seed for reproducibility. If None, uses numpy's global random state.

    Examples
    --------
    >>> splitter = RandomSplitter(test_set_ratio=0.2, random_state=42)
    >>> samples = ["sample_1", "sample_2", "sample_3", "sample_4", "sample_5"]
    >>> test_samples = splitter.split(samples)
    """

    def __init__(
        self, test_set_ratio: float = 0.2, random_state: int | None = None
    ) -> None:
        self.test_set_ratio = test_set_ratio
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def split(self, samples: Sequence[str], **kwargs: Any) -> NDArray[np.str_]:
        """Generate a random test set split.

        Parameters
        ----------
        samples
            List of sample identifiers to split.
        **kwargs
            Unused, accepted for API compatibility.

        Returns
        -------
        NDArray[np.str_]
            Array of sample identifiers for the test set.
        """
        n_test = int(len(samples) * self.test_set_ratio)
        return self._rng.choice(samples, size=n_test, replace=False)


class OutOfCladeSplitter(DataSplitter):
    """Out-of-clade (OOC) data splitter for phylogenetic validation.

    Creates test sets by selecting entire clades from a phylogenetic tree,
    ensuring that the test set contains phylogenetically distinct samples
    from the training set.

    This is useful for evaluating how well a model generalizes to
    evolutionarily distant organisms.

    Parameters
    ----------
    tree
        Phylogenetic tree (ete3.Tree object).
    test_set_range
        Min and max fraction of samples for the test set.
    single_clades
        Pre-computed list of clades. If None, computed automatically.
    n_max_clade
        Maximum number of separate clades to combine for the test set.
    prefer_small_clade
        If True, prefer selecting smaller (single) clades.
    phenotype_data
        Binary phenotype data for class balance constraints.
    min_zeros
        Minimum number of negative samples required in test set.
    min_ones
        Minimum number of positive samples required in test set.
    timeout_iterations
        Maximum iterations when searching for a valid split.
    random_state
        Random seed for reproducibility.

    Raises
    ------
    ValueError
        If min_zeros or min_ones are specified without phenotype_data.

    Examples
    --------
    >>> from ete3 import Tree
    >>> tree = Tree("((A,B),(C,D));")
    >>> splitter = OutOfCladeSplitter(tree, test_set_range=(0.2, 0.3))
    >>> test_samples = splitter.split(["A", "B", "C", "D"])
    """

    def __init__(
        self,
        tree: "Tree",
        test_set_range: tuple[float, float] = (0.2, 0.3),
        single_clades: list[list[str]] | None = None,
        n_max_clade: int = 2,
        prefer_small_clade: bool = False,
        phenotype_data: pd.Series | None = None,
        min_zeros: int = 0,
        min_ones: int = 0,
        timeout_iterations: int | None = None,
        random_state: int | None = None,
    ) -> None:
        self.tree = tree
        self.test_set_range = test_set_range
        self.single_clades = single_clades
        self.n_max_clade = n_max_clade
        self.prefer_small_clade = prefer_small_clade
        self.phenotype_data = phenotype_data
        self.min_zeros = min_zeros
        self.min_ones = min_ones
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        if (min_ones > 0 or min_zeros > 0) and phenotype_data is None:
            raise ValueError(
                "phenotype_data is required when min_zeros or min_ones > 0"
            )

        if timeout_iterations is None:
            self.timeout_iterations = len(tree.get_leaves()) * 3
        else:
            self.timeout_iterations = timeout_iterations

    def split(self, samples: Sequence[str], **kwargs: Any) -> NDArray[np.str_]:
        """Generate an out-of-clade test set split.

        Parameters
        ----------
        samples
            List of sample identifiers to split.
        **kwargs
            Unused, accepted for API compatibility.

        Returns
        -------
        NDArray[np.str_]
            Array of sample identifiers for the test set.

        Raises
        ------
        ValueError
            If no valid split is found within timeout_iterations.
        """
        if self.single_clades is None:
            self.single_clades = self._compute_single_clades(samples)

        for _ in range(self.timeout_iterations):
            if self.prefer_small_clade:
                n_clades = self._rng.integers(1, self.n_max_clade + 1)
                clades = self._rng.choice(
                    self.single_clades[1:], size=n_clades, replace=False
                ).tolist()
            else:
                clades = self._rng.choice(
                    self.single_clades, size=self.n_max_clade, replace=False
                ).tolist()

            test_samples = reduce(np.union1d, clades)

            if self._is_valid_split(test_samples, samples):
                return test_samples

        raise ValueError(
            f"Could not find valid split within {self.timeout_iterations} iterations"
        )

    def _compute_single_clades(self, samples: Sequence[str]) -> list[list[str]]:
        """Compute list of clades within the size range.

        Parameters
        ----------
        samples
            List of sample identifiers.

        Returns
        -------
        list[list[str]]
            List of clades, each represented as a list of sample names.
        """
        min_size = int(self.test_set_range[0] * len(samples))
        max_size = int(self.test_set_range[1] * len(samples))

        tree = self.tree.copy()
        tree.prune(samples, preserve_branch_length=True)

        single_clades: list[list[str]] = [[]]  # Include empty clade
        for node in tree.traverse():
            if node.is_leaf():
                continue
            leaves = [leaf.name for leaf in node.get_leaves()]
            if len(leaves) <= max_size:
                single_clades.append(leaves)

        return single_clades

    def _is_valid_split(
        self, test_samples: NDArray[np.str_], all_samples: Sequence[str]
    ) -> bool:
        """Check if a split meets size and class balance constraints.

        Parameters
        ----------
        test_samples
            Proposed test set samples.
        all_samples
            All available samples.

        Returns
        -------
        bool
            True if the split is valid.
        """
        ratio = len(test_samples) / len(all_samples)
        if ratio < self.test_set_range[0] or ratio > self.test_set_range[1]:
            return False

        if self.phenotype_data is None:
            return True

        test_phenotypes = self.phenotype_data.loc[test_samples]
        n_zeros = (test_phenotypes == 0).sum()
        n_ones = (test_phenotypes == 1).sum()

        if self.min_zeros > 0 and n_zeros < self.min_zeros:
            return False
        if self.min_ones > 0 and n_ones < self.min_ones:
            return False

        return True


class InCladeSplitter(DataSplitter):
    """In-clade data splitter for phylogenetic validation.

    Creates test sets by sampling proportionally from phylogenetic clusters,
    ensuring that test samples are distributed across the tree similarly
    to training samples.

    This is useful for evaluating model performance when test samples
    have close relatives in the training set.

    Parameters
    ----------
    tree
        Phylogenetic tree (ete3.Tree object).
    distance_df
        Pairwise distance matrix between samples (DataFrame with sample IDs
        as both index and columns).
    test_set_ratio
        Fraction of samples from each cluster to include in test set.
    random_state
        Random seed for reproducibility.

    Examples
    --------
    >>> splitter = InCladeSplitter(tree, distance_df, test_set_ratio=0.2)
    >>> test_samples = splitter.split(samples)
    """

    def __init__(
        self,
        tree: "Tree",
        distance_df: pd.DataFrame,
        test_set_ratio: float = 0.2,
        random_state: int | None = None,
    ) -> None:
        self.tree = tree
        self.distance_df = distance_df
        self.test_set_ratio = test_set_ratio
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def split(self, samples: Sequence[str], **kwargs: Any) -> NDArray[np.str_]:
        """Generate an in-clade test set split.

        Uses hierarchical clustering to group samples, then samples
        proportionally from each cluster.

        Parameters
        ----------
        samples
            List of sample identifiers to split.
        **kwargs
            Unused, accepted for API compatibility.

        Returns
        -------
        NDArray[np.str_]
            Array of sample identifiers for the test set.
        """
        samples_list = list(samples)
        distance_subset = self.distance_df.loc[samples_list, samples_list]

        # Modified square root rule: 2 * sqrt(n/2)
        n_clusters = int(2 * np.sqrt(len(samples_list) / 2))
        n_clusters = max(2, n_clusters)  # Ensure at least 2 clusters

        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="precomputed", linkage="average"
        )
        clustering.fit(distance_subset)
        labels = clustering.labels_

        test_samples: list[str] = []
        for label in np.unique(labels):
            cluster_mask = labels == label
            cluster_samples = [s for s, m in zip(samples_list, cluster_mask) if m]

            n_test = max(1, int(len(cluster_samples) * self.test_set_ratio))
            selected = self._rng.choice(
                cluster_samples, size=min(n_test, len(cluster_samples)), replace=False
            )
            test_samples.extend(selected)

        return np.array(test_samples)


# Alias for backwards compatibility
LargeTreeTraverseOOCSplitter = OutOfCladeSplitter
