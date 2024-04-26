"""Module that defines the PhenotypeSet class"""

import pathlib
from collections.abc import Iterable, Sequence
from itertools import islice
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

    def __repr__(self) -> str:
        return f"PhenotypeSet (n={len(self._phenotype_dict)})"

    def __getitem__(self, index: PhenotypeIndex) -> Phenotype:
        return self._phenotype_dict[index]

    def __len__(self) -> int:
        return len(self._phenotype_dict)

    def __iter__(self) -> Iterator[Phenotype]:
        yield from self._phenotype_dict.values()

    # NOTE: __contains__, __reversed__, index and count methods are mixins

    @staticmethod
    def make_index(name: str, category: str) -> PhenotypeIndex:
        """
        Creates a PhenotypeIndex object.

        Parameters
        ---------
        name : str
            Name of the phenotype.
        category : str
            Category of the phenotype.

        Returns
        ------
        PhenotypeIndex
            PhenotypeIndex object.
        """
        return PhenotypeIndex(name=name, category=category)

    @property
    def phenotypes(self) -> Iterable[Phenotype]:
        """Iterable of Phenotype objects."""
        return self.__iter__()

    @staticmethod
    def _parse_phenotype_info(phenotype_str: str) -> tuple[str, str]:
        """Parse the phenotype name and category from a string.

        Parameters
        ----------
        phenotype_str : str
            The string containing the phenotype name and category.

        Returns
        -------
        tuple[str, str]
            Phenotype name and category.
        """
        try:
            name, category = phenotype_str.split("--")
            if not category.strip():
                category = "unknown"
        except ValueError:
            if phenotype_str.startswith("Unnamed"):
                num = phenotype_str.rsplit(" ", 1)[-1]
                name = f"unnamed_{num}"
            else:
                name = phenotype_str
            category = "unknown"
        return name.strip(), category.strip()

    @classmethod
    def read_data(cls, file_path: str | pathlib.Path) -> "PhenotypeSet":
        """
        Reads phenotype data from a TSV file and returns a PhenotypeSet object.

        Parameters
        ---------
        file_path : str | pahtlib.Path
            Path to the TSV file containing the phenotype data.

        Returns
        ------
        "PhenotypeSet"
            PhenotypeSet object containing the phenotype data.

        """
        raw_phenotype_df = pd.read_csv(
            file_path, sep="\t", index_col=0, dtype={"genomeID": str}
        ).astype("Int64")
        if raw_phenotype_df.index.name != "genomeID":
            raise ValueError("The index of the PhenotypeSet table must be 'genomeID'")
        phenotype_df = raw_phenotype_df.select_dtypes(include="number")
        phenotypes = []
        for col in phenotype_df.columns:
            name, category = cls._parse_phenotype_info(col)
            phenotypes.append(Phenotype(phenotype_df.loc[:, col], name, category))
        return PhenotypeSet(phenotypes)

    @classmethod
    def read_files(cls, file_paths: list[str | pathlib.Path]) -> "PhenotypeSet":
        """Reads phenotype data from multiple TSV files and returns a PhenotypeSet object.

        Parameters
        ----------
        file_paths : list[str | pathlib.Path]
            List of paths to the TSV files containing the phenotype data.

        Returns
        -------
        "PhenotypeSet"
            PhenotypeSet object containing the phenotype data.
        """
        phenotypes = []
        for file_path in file_paths:
            phenotypes.extend(cls.read_data(file_path).phenotypes)
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
