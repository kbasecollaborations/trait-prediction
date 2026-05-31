"""Tests for the data splitters module."""

import numpy as np
import pandas as pd
import pytest

from trait_prediction.pipeline.splitters import (
    DataSplitter,
    InCladeSplitter,
    OutOfCladeSplitter,
    RandomSplitter,
)


class TestRandomSplitter:
    """Tests for RandomSplitter."""

    def test_split_basic(self):
        """Test basic random split functionality."""
        splitter = RandomSplitter(test_set_ratio=0.2, random_state=42)
        samples = [f"sample_{i}" for i in range(100)]

        test_samples = splitter.split(samples)

        assert len(test_samples) == 20
        assert all(s in samples for s in test_samples)
        # Check no duplicates
        assert len(set(test_samples)) == len(test_samples)

    def test_split_reproducibility(self):
        """Test that random state ensures reproducibility."""
        samples = [f"sample_{i}" for i in range(100)]

        splitter1 = RandomSplitter(test_set_ratio=0.2, random_state=42)
        splitter2 = RandomSplitter(test_set_ratio=0.2, random_state=42)

        test1 = splitter1.split(samples)
        test2 = splitter2.split(samples)

        np.testing.assert_array_equal(test1, test2)

    def test_split_different_ratios(self):
        """Test different test set ratios."""
        samples = [f"sample_{i}" for i in range(100)]

        for ratio in [0.1, 0.2, 0.3, 0.5]:
            splitter = RandomSplitter(test_set_ratio=ratio, random_state=42)
            test_samples = splitter.split(samples)
            expected_size = int(len(samples) * ratio)
            assert len(test_samples) == expected_size

    def test_generate_splits(self):
        """Test generating multiple splits."""
        splitter = RandomSplitter(test_set_ratio=0.2, random_state=42)
        samples = [f"sample_{i}" for i in range(100)]

        splits = splitter.generate_splits(samples, n=5)

        assert len(splits) == 5
        for split in splits:
            assert len(split) == 20


class TestOutOfCladeSplitter:
    """Tests for OutOfCladeSplitter (requires ete3)."""

    @pytest.fixture
    def simple_tree(self):
        """Create a simple tree for testing."""
        try:
            from ete3 import Tree
            # Create a tree with 20 leaves
            newick = "((((A1,A2),(A3,A4)),((B1,B2),(B3,B4))),(((C1,C2),(C3,C4)),((D1,D2),(D3,D4))));"
            return Tree(newick)
        except ImportError:
            pytest.skip("ete3 not installed")

    @pytest.fixture
    def tree_samples(self):
        """Get sample names from the simple tree."""
        return [f"{letter}{num}" for letter in "ABCD" for num in "1234"]

    def test_split_basic(self, simple_tree, tree_samples):
        """Test basic out-of-clade split."""
        splitter = OutOfCladeSplitter(
            tree=simple_tree,
            test_set_range=(0.2, 0.4),
            random_state=42,
        )

        test_samples = splitter.split(tree_samples)

        # Check test set size is within range
        ratio = len(test_samples) / len(tree_samples)
        assert 0.2 <= ratio <= 0.4

        # Check all test samples are valid
        assert all(s in tree_samples for s in test_samples)

    def test_split_with_phenotype_constraints(self, simple_tree, tree_samples):
        """Test split with min_zeros and min_ones constraints."""
        # Create phenotype data with balanced classes
        phenotype_data = pd.Series(
            [0, 0, 1, 1] * 4, index=tree_samples
        )

        splitter = OutOfCladeSplitter(
            tree=simple_tree,
            test_set_range=(0.2, 0.5),
            phenotype_data=phenotype_data,
            min_zeros=1,
            min_ones=1,
            random_state=42,
        )

        test_samples = splitter.split(tree_samples)

        # Check constraints
        test_phenotypes = phenotype_data.loc[test_samples]
        assert (test_phenotypes == 0).sum() >= 1
        assert (test_phenotypes == 1).sum() >= 1

    def test_split_raises_without_phenotype_data(self, simple_tree):
        """Test that min_zeros/min_ones without phenotype_data raises error."""
        with pytest.raises(ValueError, match="phenotype_data is required"):
            OutOfCladeSplitter(
                tree=simple_tree,
                min_zeros=1,
            )


class TestInCladeSplitter:
    """Tests for InCladeSplitter."""

    @pytest.fixture
    def distance_matrix(self):
        """Create a simple distance matrix for testing."""
        samples = [f"sample_{i}" for i in range(20)]
        # Create a distance matrix based on index differences
        n = len(samples)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = abs(i - j)
        return pd.DataFrame(distances, index=samples, columns=samples)

    @pytest.fixture
    def mock_tree(self):
        """Create a mock tree (not actually used in InCladeSplitter logic)."""
        try:
            from ete3 import Tree
            return Tree("((A,B),(C,D));")
        except ImportError:
            # Return a mock object since InCladeSplitter doesn't really use the tree
            return None

    def test_split_basic(self, mock_tree, distance_matrix):
        """Test basic in-clade split."""
        if mock_tree is None:
            pytest.skip("ete3 not installed")

        samples = list(distance_matrix.index)
        splitter = InCladeSplitter(
            tree=mock_tree,
            distance_df=distance_matrix,
            test_set_ratio=0.2,
            random_state=42,
        )

        test_samples = splitter.split(samples)

        # Check test set is non-empty
        assert len(test_samples) > 0

        # Check all test samples are valid
        assert all(s in samples for s in test_samples)

    def test_split_reproducibility(self, mock_tree, distance_matrix):
        """Test that random state ensures reproducibility."""
        if mock_tree is None:
            pytest.skip("ete3 not installed")

        samples = list(distance_matrix.index)

        splitter1 = InCladeSplitter(
            tree=mock_tree,
            distance_df=distance_matrix,
            test_set_ratio=0.2,
            random_state=42,
        )
        splitter2 = InCladeSplitter(
            tree=mock_tree,
            distance_df=distance_matrix,
            test_set_ratio=0.2,
            random_state=42,
        )

        test1 = splitter1.split(samples)
        test2 = splitter2.split(samples)

        np.testing.assert_array_equal(sorted(test1), sorted(test2))
