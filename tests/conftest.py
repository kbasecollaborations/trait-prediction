import random
from pathlib import Path

import pytest
from catboost import CatBoostClassifier

from trait_prediction.main import (
    DataSet,
    FeatureIndex,
    FeatureInput,
    FeatureSet,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)
from trait_prediction.training import Predictor


@pytest.fixture(scope="session")
def data_path():
    data_path = Path("tests/data")
    return data_path


@pytest.fixture(scope="session")
def hydra_path():
    data_path = Path("data")
    return data_path


@pytest.fixture
def default_config_path(data_path):
    config_path = data_path / "configs"
    return config_path


@pytest.fixture
def leaf_phenotype_folder(data_path):
    leaf_phenotype_data = data_path / "phenotypes/leaf"
    return leaf_phenotype_data


@pytest.fixture
def leaf_feature_folder(data_path):
    leaf_feature_data = data_path / "features/leaf"
    return leaf_feature_data


@pytest.fixture
def leaf_phenotype_pinputs(leaf_phenotype_folder):
    pinputs = []
    for phenotype_file in leaf_phenotype_folder.iterdir():
        name = phenotype_file.stem
        category = phenotype_file.parent.stem
        pindex = PhenotypeIndex(name=name, category=category)
        index_format_func = lambda x: x
        pinput = PhenotypeInput(
            path=phenotype_file, pindex=pindex, index_format_func=index_format_func
        )
        pinputs.append(pinput)
    return pinputs


@pytest.fixture
def leaf_phenotype(leaf_phenotype_pinputs):
    phenotype_set = PhenotypeSet.read_data(leaf_phenotype_pinputs)
    return random.choice(list(phenotype_set.phenotypes))


@pytest.fixture(scope="function")
def leaf_phenotype_set(leaf_phenotype_pinputs):
    return PhenotypeSet.read_data(leaf_phenotype_pinputs)


@pytest.fixture
def leaf_feature(leaf_feature_finputs):
    feature_set = FeatureSet.read_data(leaf_feature_finputs)
    return random.choice(list(feature_set.features))


@pytest.fixture(scope="function")
def leaf_feature_set(leaf_feature_finputs):
    return FeatureSet.read_data(leaf_feature_finputs)


@pytest.fixture(scope="function")
def leaf_dataset(leaf_phenotype_pinputs, leaf_feature_finputs):
    dataset = DataSet.read_data(leaf_phenotype_pinputs, leaf_feature_finputs)
    return dataset


@pytest.fixture
def leaf_dataset_data(leaf_dataset):
    dataset = leaf_dataset
    pindex = random.choice([p.pindex for p in dataset.phenotypes])
    findex = random.choice([f.findex for f in dataset.features])
    return dataset.get_data(pindex, findex)


@pytest.fixture
def leaf_feature_finputs(leaf_feature_folder):
    finputs = []
    for feature_file in leaf_feature_folder.iterdir():
        name = feature_file.stem
        ftype = "binary"
        dtype = "uint8"
        findex = FeatureIndex(name=name, ftype=ftype, dtype=dtype)
        index_format_func = lambda x: x
        finput = FeatureInput(
            path=feature_file, findex=findex, index_format_func=index_format_func
        )
        finputs.append(finput)
    return finputs


@pytest.fixture
def random_state():
    return 42


@pytest.fixture(scope="function")
def leaf_predictor(leaf_dataset_data, random_state):
    phenotype, feature = leaf_dataset_data
    classifier = CatBoostClassifier(
        random_state=random_state,
        objective="Logloss",
        verbose=False,
        allow_writing_files=False,
        thread_count=1,
    )
    predictor = Predictor(phenotype, feature, classifier, random_state=random_state)
    return predictor
