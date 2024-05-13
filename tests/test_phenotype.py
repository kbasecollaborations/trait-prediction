from trait_prediction.main import (
    Phenotype,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)


def test_phenotype_read_data(leaf_phenotype_folder):
    for phenotype_file in leaf_phenotype_folder.iterdir():
        name = phenotype_file.stem
        category = phenotype_file.parent.stem
        pindex = PhenotypeIndex(name=name, category=category)
        index_format_func = lambda x: x
        pinput = PhenotypeInput(
            path=phenotype_file, pindex=pindex, index_format_func=index_format_func
        )
        phenotype = Phenotype.read_data(pinput)
        assert phenotype.phenotype_data is not None
        assert phenotype.phenotype_data.shape[0] > 0
        assert phenotype.pindex.name == phenotype_file.stem
        assert phenotype.pindex.category == phenotype_file.parent.stem


def test_phenotypeset_read_data(leaf_phenotype_pinputs):
    phenotype_set = PhenotypeSet.read_data(leaf_phenotype_pinputs)
    assert len(phenotype_set) == len(leaf_phenotype_pinputs)


def test_phenotypeset_get_phenotype(leaf_phenotype_pinputs):
    phenotype_set = PhenotypeSet.read_data(leaf_phenotype_pinputs)
    pindex = PhenotypeIndex(name="histidine", category="leaf")
    phenotype = phenotype_set.get_phenotype(pindex)
    assert phenotype.pindex.name == "histidine"
