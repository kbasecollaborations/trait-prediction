"""Module that defines the PhenotypeSet class"""

from collections.abc import Sequence
from itertools import islice
from typing import Iterable, Iterator

from .phenotype import Phenotype


class PhenotypeSet(Sequence[Phenotype]):
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
        self._phenotype_dict = {
            PhenotypeIndex(name=phenotype.name, category=phenotype.category): phenotype
            for phenotype in phenotypes
        }

    def __repr__(self) -> str:
        return f"PhenotypeSet (n={len(self._phenotype_dict)})"

    def __getitem__(self, index):
        return list(self._phenotype_dict.values())[index]

    def __len__(self) -> int:
        return len(self._phenotype_dict)

    def __iter__(self) -> Iterator[Phenotype]:
        yield from self._phenotype_dict.values()

    # NOTE: __contains__, __reversed__, index and count methods are mixins

    def get_phenotype(self, name: str, category: str) -> Phenotype:
        """
        Get a phenotype by name and category.

        Parameters
        ---------
        name : str
            Name of the phenotype.
        category : str
            Category of the phenotype.

        Returns
        ------
        Phenotype
            Phenotype object.
        """
        return self._phenotype_dict[PhenotypeIndex.make_index(name, category)]

    @property
    def phenotypes(self) -> Iterable[Phenotype]:
        """Iterable of Phenotype objects."""
        return self.__iter__()

    @classmethod
    def read_data(
        cls,
        inputs: list[PhenotypeInput],
    ) -> "PhenotypeSet":
        """Reads phenotype data from multiple TSV files and returns a PhenotypeSet object.

        Parameters
        ----------
        inputs : list[PhenotypeInput]
            List of phenotype input data.

        Returns
        -------
        "PhenotypeSet"
            PhenotypeSet object containing the phenotype data.
        """
        phenotypes = []
        for phenotype_input in inputs:
            file_path = phenotype_input.path
            category = phenotype_input.category
            phenotypes.append(Phenotype.read_data(file_path, category))
        return cls(phenotypes)

    @classmethod
    def limit(cls, phenotype_set: "PhenotypeSet", limit: int) -> "PhenotypeSet":
        """
        Create a new PhenotypeSet object with a limited number of phenotypes.

        Parameters
        ---------
        phenotype_set : "PhenotypeSet"
            The PhenotypeSet object to limit
        limit : int
            The number of phenotypes to limit to

        Returns
        ------
        "PhenotypeSet"
            The PhenotypeSet object with limited number of phenotypes
        """
        limited_phenotypes = list(islice(phenotype_set.phenotypes, limit))
        return cls(limited_phenotypes)
