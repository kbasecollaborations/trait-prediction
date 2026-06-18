"""Shared fixtures for the minimal test suite.

Fixtures are intentionally lean and built on the small ``tests/data/leaf``
dataset (3 phenotypes x 2 feature matrices) so the whole suite stays fast.
"""

import random
from pathlib import Path

import pytest

from trait_prediction.main import (
    DataSet,
    FeatureIndex,
    FeatureInput,
    FeatureSet,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)


@pytest.fixture(scope="session")
def data_path():
    return Path("tests/data")


@pytest.fixture
def config_path(data_path):
    return data_path / "configs"


def _read_pinputs(folder):
    pinputs = []
    for phenotype_file in folder.iterdir():
        pindex = PhenotypeIndex(name=phenotype_file.stem, category=folder.stem)
        pinputs.append(
            PhenotypeInput(
                path=phenotype_file, pindex=pindex, index_format_func=lambda x: x
            )
        )
    return pinputs


def _read_finputs(folder):
    finputs = []
    for feature_file in folder.iterdir():
        findex = FeatureIndex(name=feature_file.stem, ftype="binary", dtype="uint8")
        finputs.append(
            FeatureInput(
                path=feature_file, findex=findex, index_format_func=lambda x: x
            )
        )
    return finputs


@pytest.fixture
def phenotype_pinputs(data_path):
    return _read_pinputs(data_path / "phenotypes/leaf")


@pytest.fixture
def feature_finputs(data_path):
    return _read_finputs(data_path / "features/leaf")


@pytest.fixture
def ch_phenotype_pinputs(data_path):
    return _read_pinputs(data_path / "phenotypes/ch")


@pytest.fixture
def ch_feature_finputs(data_path):
    return _read_finputs(data_path / "features/ch")


@pytest.fixture
def phenotype_set(phenotype_pinputs):
    return PhenotypeSet.read_data(phenotype_pinputs)


@pytest.fixture
def feature_set(feature_finputs):
    return FeatureSet.read_data(feature_finputs)


@pytest.fixture
def dataset(phenotype_pinputs, feature_finputs):
    return DataSet.read_data(phenotype_pinputs, feature_finputs)


@pytest.fixture
def dataset_data(dataset):
    """A single aligned (phenotype, feature) pair drawn from the dataset."""
    pindex = random.choice([p.pindex for p in dataset.phenotypes])
    findex = random.choice([f.findex for f in dataset.features])
    return dataset.get_data(pindex, findex)
