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
