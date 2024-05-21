"""Module that defines the FeatureSet class"""

from collections.abc import Set
from typing import Iterable, Iterator

from .feature import Feature, FeatureIndex, FeatureInput


class FeatureSet(Set[Feature]):
    """Class that represents a collection of features.

    Parameters
    ---------
    features : Iterable[Feature]
        List of Feature objects.

    Attributes
    ---------
    features : list[Feature]
        List of Feature objects.
    """

    def __init__(self, features: list[Feature]) -> None:
        super().__init__()
        self._feature_dict = {feature.findex: feature for feature in features}

    def __repr__(self) -> str:
        return f"FeatureSet (n={len(self._feature_dict)})"

    def __getitem__(self, index):
        return list(self._feature_dict.values())[index]

    def __len__(self) -> int:
        return len(self._feature_dict)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._feature_dict.values())

    def __contains__(self, feature: Feature) -> bool:  # type: ignore
        return feature.findex in self._feature_dict

    def get_feature(self, findex: FeatureIndex) -> Feature:
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
        return self._feature_dict[findex]

    @property
    def features(self) -> Iterable[Feature]:
        """Iterable of Feature objects."""
        return self.__iter__()

    @classmethod
    def read_data(cls, finputs: list[FeatureInput]) -> "FeatureSet":
        """
        Reads the feature data from the given inputs and returns a FeatureSet object.

        Parameters
        ---------
        inputs : list[FeatureInput]
            List of FeatureInput objects.

        Returns
        ------
        FeatureSet
            FeatureSet object.
        """
        features = []
        for finput in finputs:
            features.append(Feature.read_data(finput))
        return cls(features)

    @classmethod
    def merge_data(cls, finput_tuples: list[tuple[FeatureInput, ...]]) -> "FeatureSet":
        """
        Merges the feature data from multiple Feature objects and returns a FeatureSet object.

        Parameters
        ---------
        finput_tuples : list[tuple[FeatureInput, ...]]
            List of tuples containing FeatureInput objects.

        Returns
        ------
        FeatureSet
            FeatureSet object containing the merged feature data.
        """
        features = []
        for finputs in finput_tuples:
            features.append(Feature.merge_data(finputs))
        return cls(features)
