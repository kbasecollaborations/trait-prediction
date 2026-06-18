"""Phylogeny-aware train/test splitters.

The random / out-of-clade / in-clade splitting regimes are central to the
manuscript's study of cross-dataset generalisation.
"""

import numpy as np
import pandas as pd
import pytest

from trait_prediction.pipeline.splitters import (
    InCladeSplitter,
    OutOfCladeSplitter,
    RandomSplitter,
)

SAMPLES = [f"{letter}{num}" for letter in "ABCD" for num in "1234"]
NEWICK = "((((A1,A2),(A3,A4)),((B1,B2),(B3,B4))),(((C1,C2),(C3,C4)),((D1,D2),(D3,D4))));"


def test_random_splitter_size_and_reproducibility():
    splitter = RandomSplitter(test_set_ratio=0.25, random_state=42)
    test1 = splitter.split(SAMPLES)
    test2 = RandomSplitter(test_set_ratio=0.25, random_state=42).split(SAMPLES)

    assert len(test1) == int(len(SAMPLES) * 0.25)
    assert set(test1).issubset(SAMPLES)
    np.testing.assert_array_equal(test1, test2)


def test_random_splitter_generate_splits():
    splitter = RandomSplitter(test_set_ratio=0.25, random_state=42)
    splits = splitter.generate_splits(SAMPLES, n=3)
    assert len(splits) == 3
    assert all(len(s) == int(len(SAMPLES) * 0.25) for s in splits)


def test_out_of_clade_splitter_respects_range():
    tree = pytest.importorskip("ete3").Tree(NEWICK)
    splitter = OutOfCladeSplitter(
        tree=tree, test_set_range=(0.2, 0.4), random_state=42
    )
    test_samples = splitter.split(SAMPLES)
    ratio = len(test_samples) / len(SAMPLES)
    assert 0.2 <= ratio <= 0.4
    assert set(test_samples).issubset(SAMPLES)


def test_in_clade_splitter_basic():
    tree = pytest.importorskip("ete3").Tree(NEWICK)
    n = len(SAMPLES)
    distances = pd.DataFrame(
        np.abs(np.subtract.outer(range(n), range(n))),
        index=SAMPLES,
        columns=SAMPLES,
    )
    splitter = InCladeSplitter(
        tree=tree, distance_df=distances, test_set_ratio=0.25, random_state=42
    )
    test_samples = splitter.split(SAMPLES)
    assert len(test_samples) > 0
    assert set(test_samples).issubset(SAMPLES)
