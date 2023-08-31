"""Module that defines the PhenotypeSet class"""

from collections.abc import Iterable, Sequence
from typing import Iterator, NamedTuple

import pandas as pd

from .phenotype import Phenotype


class PhenotypeIndex(NamedTuple):
    """Class that represents a phenotype index."""

    name: str
    category: str


class PhenotypeSet(Sequence[Phenotype]):
    """
    Class that represents a collection of phenotypes.

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

    def __getitem__(self, index: PhenotypeIndex) -> Phenotype:
        return self._phenotype_dict[index]

    def __len__(self) -> int:
        return len(self._phenotype_dict)

    def __iter__(self) -> Iterator[Phenotype]:
        for phenotype in self._phenotype_dict.values():
            yield phenotype

    # NOTE: __contains__, __reversed__, index and count methods are mixins

    @property
    def phenotypes(self) -> Iterable[Phenotype]:
        """Iterable of Phenotype objects."""
        return self.__iter__()

    @classmethod
    def read_data(cls, file_path: str) -> "PhenotypeSet":
        """
        Reads phenotype data from a TSV file and returns a PhenotypeSet object.

        Parameters
        ---------
        data : str
            Path to the TSV file containing the phenotype data.

        Returns
        ------
        "PhenotypeSet"
            PhenotypeSet object containing the phenotype data.

        """
        raw_phenotype_df = pd.read_csv(file_path, sep="\t", index_col=0)
        phenotype_df = raw_phenotype_df.select_dtypes(include="number")
        phenotypes = []
        for col in phenotype_df.columns:
            try:
                name, category = col.split("--")
                if not category:
                    category = "unknown"
            except ValueError:
                if col.startswith("Unnamed"):
                    num = col.rsplit(" ", 1)[-1]
                    name = f"unnamed_{num}"
                    category = "unknown"
                else:
                    name = col
                    category = "unknown"
            phenotypes.append(Phenotype(phenotype_df[col], name, category))
        return PhenotypeSet(phenotypes)
