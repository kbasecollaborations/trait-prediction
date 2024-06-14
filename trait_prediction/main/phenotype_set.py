"""Module that defines the PhenotypeSet class"""

from collections.abc import Set
from itertools import islice
from typing import Iterable, Iterator

from .phenotype import Phenotype, PhenotypeIndex, PhenotypeInput


class PhenotypeSet(Set[Phenotype]):
    """Class that represents a collection of phenotypes.

    Parameters
    ---------
    phenotypes : Iterable[Phenotype]
        List of Phenotype objects.

    Attributes
    ---------
    phenotypes : list[Phenotype]
        List of Phenotype objects.
    """

    def __init__(self, phenotypes: list[Phenotype]) -> None:
        super().__init__()
        self._phenotype_dict = {phenotype.pindex: phenotype for phenotype in phenotypes}

    def __repr__(self) -> str:
        return f"PhenotypeSet (n={len(self._phenotype_dict)})"

    def __getitem__(self, index):
        return list(self._phenotype_dict.values())[index]

    def __len__(self) -> int:
        return len(self._phenotype_dict)

    def __iter__(self) -> Iterator[Phenotype]:
        yield from self._phenotype_dict.values()

    def __contains__(self, phenotype: Phenotype) -> bool:  # type: ignore
        return phenotype.pindex in self._phenotype_dict

    def get_phenotype(self, pindex: PhenotypeIndex) -> Phenotype:
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
        return self._phenotype_dict[pindex]

    @property
    def phenotypes(self) -> Iterable[Phenotype]:
        """Iterable of Phenotype objects."""
        return self.__iter__()

    @classmethod
    def read_data(
        cls,
        pinputs: list[PhenotypeInput],
    ) -> "PhenotypeSet":
        """Reads phenotype data from multiple TSV files and returns a PhenotypeSet object.

        Parameters
        ----------
        pinputs : list[PhenotypeInput]
            List of phenotype input data.

        Returns
        -------
        "PhenotypeSet"
            PhenotypeSet object containing the phenotype data.
        """
        phenotypes = []
        for pinput in pinputs:
            phenotypes.append(Phenotype.read_data(pinput))
        return cls(phenotypes)

    @classmethod
    def merge_data(
        cls, pinput_tuples: list[tuple[PhenotypeInput, ...]]
    ) -> "PhenotypeSet":
        """
        Merge grouped PhenotypeInput objects and return a PhenotypeSet object.

        Parameters
        ---------
        pinput_tuples : list[tuple[PhenotypeInput, ...]]
            List of tuples containing multiple PhenotypeInput objects.

        Returns
        ------
        "PhenotypeSet"
            PhenotypeSet object containing the merged phenotype data.
        """
        phenotypes = []
        for pinput_tuple in pinput_tuples:
            phenotypes.append(Phenotype.merge_data(pinput_tuple))
        return cls(phenotypes)

    def limit(self, limit: int) -> "PhenotypeSet":
        """
        Create a new PhenotypeSet object with a limited number of phenotypes.

        Parameters
        ---------
        limit : int
            The number of phenotypes to limit to

        Returns
        ------
        "PhenotypeSet"
            The PhenotypeSet object with limited number of phenotypes
        """
        limited_phenotypes = list(islice(self.phenotypes, limit))
        return PhenotypeSet(limited_phenotypes)
