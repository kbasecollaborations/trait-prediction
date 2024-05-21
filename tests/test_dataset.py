from collections import defaultdict

from trait_prediction.main import DataSet, FeatureIndex, PhenotypeIndex


def test_dataset_read_data(leaf_phenotype_pinputs, leaf_feature_finputs):
    dataset = DataSet.read_data(leaf_phenotype_pinputs, leaf_feature_finputs)
    assert len(list(dataset.features)) == len(leaf_feature_finputs)
    assert len(list(dataset.phenotypes)) == len(leaf_phenotype_pinputs)


def test_dataset_get_phenotype(leaf_dataset):
    dataset = leaf_dataset
    pindex = PhenotypeIndex(name="histidine", category="leaf")
    phenotype = dataset.get_phenotype(pindex)
    assert phenotype.pindex.name == "histidine"


def test_dataset_get_feature(leaf_dataset):
    dataset = leaf_dataset
    findex = FeatureIndex(name="kofam_20", ftype="binary", dtype="uint8")
    feature = dataset.get_feature(findex)
    assert feature.findex.name == "kofam_20"


def test_dataset_get_data(leaf_dataset):
    dataset = leaf_dataset
    pindex = PhenotypeIndex(name="histidine", category="leaf")
    findex = FeatureIndex(name="kofam_20", ftype="binary", dtype="uint8")
    phenotype, feature = dataset.get_data(pindex, findex)
    assert phenotype.phenotype_data is not None
    assert feature.feature_data is not None
    assert phenotype.phenotype_data.shape[0] == feature.feature_data.shape[0]
    assert (phenotype.phenotype_data.index == feature.feature_data.index).all()


def test_dataset_merge(
    leaf_phenotype_pinputs,
    ch_phenotype_pinputs,
    leaf_feature_finputs,
    ch_feature_finputs,
):
    phenotype_groups = defaultdict(list)
    feature_groups = defaultdict(list)
    for pinput in leaf_phenotype_pinputs + ch_phenotype_pinputs:
        phenotype_groups[pinput.pindex.name].append(pinput)
    for finput in leaf_feature_finputs + ch_feature_finputs:
        feature_groups[finput.findex.name].append(finput)
    pinput_tuples = []
    finput_tuples = []
    for _, phenotype_group in phenotype_groups.items():
        phenotype_group_list = tuple(phenotype_group)
        if len(phenotype_group_list) > 1:
            pinput_tuples.append(phenotype_group_list)
    for _, feature_group in feature_groups.items():
        feature_group_list = tuple(feature_group)
        if len(feature_group_list) > 1:
            finput_tuples.append(feature_group_list)
    dataset = DataSet.merge_data(pinput_tuples, finput_tuples)
    dataset_leaf = DataSet.read_data(leaf_phenotype_pinputs, leaf_feature_finputs)
    phenotype_names = {p.pindex.name for p in dataset.phenotypes}
    assert phenotype_names == {"histidine", "maltose", "galacturonic_acid"}
    assert max([len(p.phenotype_data) for p in dataset.phenotypes]) > max(
        [len(p.phenotype_data) for p in dataset_leaf.phenotypes]
    )
