import pytest

from trait_prediction.main import (
    Feature,
    FeatureIndex,
    FeatureInput,
    FeatureSet,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)


def test_feature_read_data(leaf_feature_folder):
    for feature_file in leaf_feature_folder.iterdir():
        name = feature_file.stem
        ftype = "binary"
        dtype = "uint8"
        findex = FeatureIndex(name=name, ftype=ftype, dtype=dtype)
        index_format_func = lambda x: x
        finput = FeatureInput(
            path=feature_file, findex=findex, index_format_func=index_format_func
        )
        feature = Feature.read_data(finput)
        assert feature.feature_data is not None
        assert feature.feature_data.shape[0] > 0
        assert feature.feature_data.shape[1] > 0
        assert feature.findex.name == feature_file.stem


def test_featureset_read_data(leaf_feature_finputs):
    feature_set = FeatureSet.read_data(leaf_feature_finputs)
    assert len(feature_set) == len(leaf_feature_finputs)


def tests_featureset_get_feature(leaf_feature_finputs):
    feature_set = FeatureSet.read_data(leaf_feature_finputs)
    findex = FeatureIndex(name="kofam_20", ftype="binary", dtype="uint8")
    feature = feature_set.get_feature(findex)
    assert feature.findex.name == "kofam_20"


def test_feature_var(leaf_feature):
    feature = leaf_feature
    new_feature_data, removed_features = Feature.remove_features_with_low_variance(
        feature.feature_data, threshold=0.1
    )
    assert len(removed_features) >= 1
    assert new_feature_data.shape[1] == feature.feature_data.shape[1] - len(
        removed_features
    )


@pytest.mark.parametrize("parallel", [True, False])
def test_feature_correlation(parallel, leaf_feature):
    feature = leaf_feature
    new_feature_data, corr_dict = Feature.remove_features_with_high_correlation(
        feature.feature_data, threshold=0.5, parallel=parallel
    )
    assert len(corr_dict) >= 1
    assert new_feature_data.shape[1] < feature.feature_data.shape[1]


@pytest.mark.parametrize("score_func", ["f_classif", "chi2", "mutual_info_classif"])
def test_feature_kbest(score_func, leaf_dataset_data):
    phenotype, feature = leaf_dataset_data
    new_feature_data, low_score_features = Feature.feature_selection_kbest(
        feature.feature_data,
        phenotype.phenotype_data,
        score_func=score_func,
        n_features=5,
    )
    assert len(low_score_features) == 14
    assert new_feature_data.shape[1] == 5


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("method", ["PCA", "NMF"])
def test_feature_dimreduction(method, leaf_feature):
    feature = leaf_feature
    new_feature_data, components_df = Feature.feature_dimensionality_reduction(
        feature.feature_data, method=method, n_components=5
    )
    assert new_feature_data.shape[1] == 5
    assert components_df.shape[0] == 5
    assert components_df.shape[1] == feature.feature_data.shape[1]
