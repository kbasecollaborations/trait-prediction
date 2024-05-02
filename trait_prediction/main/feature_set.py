"""Module that defines the FeatureSet class"""

import pathlib
from collections.abc import Sequence
from typing import Callable, Iterable, Iterator, NamedTuple

from .feature import Feature


class FeatureIndex(NamedTuple):
    """Class that represents a feature index."""

    name: str
    ftype: str
    dtype: str

    @classmethod
    def make_index(cls, name: str, ftype: str, dtype: str) -> "FeatureIndex":
        """
        Creates a FeatureIndex object.

        Parameters
        ---------
        name : str
            Name of the feature.
        ftype : str
            ftype of the feature.
        dtype : str
            dtype of the feature.

        Returns
        ------
        FeatureIndex
            FeatureIndex object.
        """
        return cls(name=name, ftype=ftype, dtype=dtype)


class FeatureInput(NamedTuple):
    """Class that represents a feature input."""

    path: pathlib.Path | str
    name: str
    ftype: str
    dtype: str
    index_format_func: Callable[[str], str]

    @classmethod
    def make_input(
        cls,
        path: pathlib.Path | str,
        name: str,
        ftype: str,
        dtype: str,
        index_format_func: Callable[[str], str],
    ) -> "FeatureInput":
        """
        Creates a FeatureInput object.

        Parameters
        ---------
        path : pathlib.Path | str
            Path to the feature file.
        name : str
            Name of the feature.
        ftype : str
            ftype of the feature.
        dtype : str
            dtype of the feature.
        index_format_func : Callable[[str], str]
            Function to format the index of the feature data.
            Eg: lambda x: x.strip().split("?")[-1].removesuffix(".RAST").removesuffix(".fna")

        Returns
        ------
        FeatureInput
            FeatureInput object.
        """
        return cls(
            path=path,
            name=name,
            ftype=ftype,
            dtype=dtype,
            index_format_func=index_format_func,
        )


class FeatureSet(Sequence[Feature]):
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
        self._feature_dict = {
            FeatureIndex(
                name=feature.name, ftype=feature.ftype, dtype=feature.dtype
            ): feature
            for feature in features
        }

    def __repr__(self) -> str:
        return f"FeatureSet (n={len(self._feature_dict)})"

    def __getitem__(self, index):
        return list(self._feature_dict.values())[index]

    def __len__(self) -> int:
        return len(self._feature_dict)

    def __iter__(self) -> Iterator[Feature]:
        return iter(self._feature_dict.values())

    # NOTE: __contains__, __reversed__, index and count methods are mixins

    def get_feature(self, name: str, ftype: str, dtype: str) -> Feature:
        """
        Returns the Feature object with the given name, ftype and dtype.

        Parameters
        ---------
        name : str
            Name of the feature.
        ftype : str
            ftype of the feature.
        dtype : str
            dtype of the feature.

        Returns
        ------
        Feature
            Feature object.
        """
        return self._feature_dict[FeatureIndex(name=name, ftype=ftype, dtype=dtype)]

    @property
    def features(self) -> Iterable[Feature]:
        """Iterable of Feature objects."""
        return self.__iter__()

    @classmethod
    def read_data(cls, inputs: list[FeatureInput]) -> "FeatureSet":
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
        for feature_input in inputs:
            feature = Feature.read_data(
                file_path=feature_input.path,
                name=feature_input.name,
                ftype=feature_input.ftype,
                dtype=feature_input.dtype,
                index_format_func=feature_input.index_format_func,
            )
            features.append(feature)
        return cls(features)
