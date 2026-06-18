"""Core data model: phenotypes, features, and the dataset that aligns them.

Covers ``trait_prediction.main`` -- ``Phenotype``/``PhenotypeSet``,
``Feature``/``FeatureSet``, ``DataSet`` (incl. the cross-dataset
``merge_data``) -- and the Numba correlation filter (``feature_corr``). This is
the data-loading path the manuscript relies on via ``DataSet.read_data``.
"""

from collections import defaultdict

import pytest

from trait_prediction.main import (
    DataSet,
    Feature,
    FeatureIndex,
    Phenotype,
    PhenotypeIndex,
)

# --------------------------------------------------------------------------- #
# Phenotype / PhenotypeSet
# --------------------------------------------------------------------------- #


def test_phenotype_read_data(phenotype_pinputs):
    phenotype = Phenotype.read_data(phenotype_pinputs[0])
    assert phenotype.phenotype_data is not None
    assert phenotype.phenotype_data.shape[0] > 0
    assert phenotype.pindex.name == phenotype_pinputs[0].pindex.name


def test_phenotypeset_read_and_get(phenotype_set, phenotype_pinputs):
    assert len(phenotype_set) == len(phenotype_pinputs)
    pindex = PhenotypeIndex(name="histidine", category="leaf")
    assert phenotype_set.get_phenotype(pindex).pindex.name == "histidine"


def test_phenotypeset_limit(phenotype_set):
    assert len(phenotype_set.limit(2)) == 2


# --------------------------------------------------------------------------- #
# Feature / FeatureSet
# --------------------------------------------------------------------------- #


def test_feature_read_data(feature_finputs):
    feature = Feature.read_data(feature_finputs[0])
    assert feature.feature_data is not None
    assert feature.feature_data.shape[0] > 0
    assert feature.feature_data.shape[1] > 0
    assert feature.findex.name == feature_finputs[0].findex.name


def test_featureset_read_and_get(feature_set, feature_finputs):
    assert len(feature_set) == len(feature_finputs)
    findex = FeatureIndex(name="kofam_20", ftype="binary", dtype="uint8")
    assert feature_set.get_feature(findex).findex.name == "kofam_20"


# --------------------------------------------------------------------------- #
# DataSet
# --------------------------------------------------------------------------- #


def test_dataset_read_data(phenotype_pinputs, feature_finputs):
    dataset = DataSet.read_data(phenotype_pinputs, feature_finputs)
    assert len(list(dataset.phenotypes)) == len(phenotype_pinputs)
    assert len(list(dataset.features)) == len(feature_finputs)


def test_dataset_get_phenotype_and_feature(dataset):
    pindex = PhenotypeIndex(name="histidine", category="leaf")
    findex = FeatureIndex(name="kofam_20", ftype="binary", dtype="uint8")
    assert dataset.get_phenotype(pindex).pindex.name == "histidine"
    assert dataset.get_feature(findex).findex.name == "kofam_20"


def test_dataset_get_data_aligned(dataset):
    """Phenotype and feature matrices returned by get_data share an index."""
    pindex = PhenotypeIndex(name="histidine", category="leaf")
    findex = FeatureIndex(name="kofam_20", ftype="binary", dtype="uint8")
    phenotype, feature = dataset.get_data(pindex, findex)

    assert phenotype.phenotype_data.shape[0] == feature.feature_data.shape[0]
    assert (phenotype.phenotype_data.index == feature.feature_data.index).all()


def test_dataset_merge_data(
    phenotype_pinputs,
    ch_phenotype_pinputs,
    feature_finputs,
    ch_feature_finputs,
):
    """merge_data combines samples for phenotypes/features shared across datasets."""
    phenotype_groups = defaultdict(list)
    feature_groups = defaultdict(list)
    for pinput in phenotype_pinputs + ch_phenotype_pinputs:
        phenotype_groups[pinput.pindex.name].append(pinput)
    for finput in feature_finputs + ch_feature_finputs:
        feature_groups[finput.findex.name].append(finput)

    pinput_tuples = [tuple(g) for g in phenotype_groups.values() if len(g) > 1]
    finput_tuples = [tuple(g) for g in feature_groups.values() if len(g) > 1]

    merged = DataSet.merge_data(pinput_tuples, finput_tuples)
    leaf_only = DataSet.read_data(phenotype_pinputs, feature_finputs)

    # Only phenotypes present in both leaf and ch survive the merge.
    assert {p.pindex.name for p in merged.phenotypes} == {
        "histidine",
        "maltose",
        "galacturonic_acid",
    }
    # Merged phenotypes pool samples from both datasets, so they are larger.
    assert max(len(p.phenotype_data) for p in merged.phenotypes) > max(
        len(p.phenotype_data) for p in leaf_only.phenotypes
    )


# --------------------------------------------------------------------------- #
# Feature preprocessing / selection
# --------------------------------------------------------------------------- #


def test_remove_low_variance_features(dataset_data):
    _, feature = dataset_data
    filtered, removed = Feature.remove_features_with_low_variance(
        feature.feature_data, threshold=0.1
    )
    assert len(removed) >= 1
    assert filtered.shape[1] == feature.feature_data.shape[1] - len(removed)


@pytest.mark.parametrize("parallel", [True, False])
def test_remove_correlated_features(parallel, dataset_data):
    """Exercises both the serial and Numba-parallel correlation filters."""
    _, feature = dataset_data
    filtered, corr_dict = Feature.remove_features_with_high_correlation(
        feature.feature_data, threshold=0.5, parallel=parallel
    )
    assert len(corr_dict) >= 1
    assert filtered.shape[1] < feature.feature_data.shape[1]


def test_feature_selection_kbest(dataset_data):
    phenotype, feature = dataset_data
    selected, low_score = Feature.feature_selection_kbest(
        feature.feature_data,
        phenotype.phenotype_data,
        score_func="chi2",
        n_features=5,
    )
    assert selected.shape[1] == 5
    assert len(low_score) == feature.feature_data.shape[1] - 5


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", ["PCA", "NMF"])
def test_feature_dimensionality_reduction(method, dataset_data):
    _, feature = dataset_data
    reduced, components = Feature.feature_dimensionality_reduction(
        feature.feature_data, method=method, n_components=5
    )
    assert reduced.shape[1] == 5
    assert components.shape[0] == 5
    assert components.shape[1] == feature.feature_data.shape[1]
