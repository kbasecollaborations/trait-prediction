from dataclasses import dataclass

from .feature import Feature, FeatureIndex, FeatureInput
from .feature_set import FeatureSet
from .phenotype import Phenotype, PhenotypeIndex, PhenotypeInput
from .phenotype_set import PhenotypeSet

# TODO: Only match the phenotype and feature matrix that is asked for -> eg. get_xy_data method


@dataclass
class DataSet:
    """Class that represents a dataset.

    Attributes
    ----------
    feature_set : FeatureSet
        FeatureSet object.
    phenotype_set : PhenotypeSet
        PhenotypeSet object.
    """

    feature_set: FeatureSet
    phenotype_set: PhenotypeSet

    @classmethod
    def read_data(
        cls,
        finputs: list[FeatureInput],
        pinputs: list[PhenotypeInput],
    ) -> "DataSet":
        """
        Reads the feature and phenotype data from the given inputs.

        Parameters
        ---------
        finputs : list[FeatureInput]
            List of FeatureInput objects.
        pinputs : list[PhenotypeInput]
            List of PhenotypeInput objects.

        Returns
        ------
        DataSet
            DataSet object.
        """
        feature_set = FeatureSet.read_data(finputs)
        phenotype_set = PhenotypeSet.read_data(pinputs)
        return cls(feature_set, phenotype_set)

    def get_phenotype(self, pindex: PhenotypeIndex):
        """
        Get a phenotype by name and category.

        Parameters
        ---------
        pindex : PhenotypeIndex
            Phenotype index containing the name and category of the phenotype.

        Returns
        ------
        Phenotype
            Phenotype object.
        """
        return self.phenotype_set.get_phenotype(pindex)

    def get_feature(self, findex: FeatureIndex):
        """
        Returns the Feature object with the given name, ftype and dtype.

        Parameters
        ---------
        findex : FeatureIndex
            Feature index containing the name, ftype and dtype of the feature.

        Returns
        ------
        Feature
            Feature object.
        """
        return self.feature_set.get_feature(findex)

    @property
    def features(self):
        """Iterable of Feature objects."""
        return self.feature_set.features

    @property
    def phenotypes(self):
        """Iterable of Phenotype objects."""
        return self.phenotype_set.phenotypes

    def get_data(
        self, pindex: PhenotypeIndex, findex: FeatureIndex
    ) -> tuple[Phenotype, Feature]:
        """
        Retrieves the phenotype and feature data and synchronizes the indices.

        Parameters
        ---------
        pindex : PhenotypeIndex
            Phenotype index containing the name and category of the phenotype.
        findex : FeatureIndex
            Feature index containing the name, ftype and dtype of the feature.

        Returns
        ------
        Phenotype
            Phenotype object.
        Feature
            Feature object.
        """
        phenotype = self.get_phenotype(pindex)
        phenotype_data = phenotype.phenotype_data
        feature = self.get_feature(findex)
        feature_data = feature.feature_data
        common_genomes = sorted(
            list(set(phenotype_data.index).intersection(set(feature_data.index)))
        )
        phenotype_data_common = phenotype_data.loc[common_genomes]
        phenotype_common = Phenotype(phenotype_data_common, pindex)
        feature_data_common = feature_data.loc[common_genomes, :]
        feature_common = Feature(feature_data_common, findex)
        return phenotype_common, feature_common
