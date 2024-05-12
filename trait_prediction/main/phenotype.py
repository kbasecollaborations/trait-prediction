"""Module that defines the Phenotype class"""

import pathlib
import pickle
from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass
class PhenotypeIndex:
    """Class that represents a phenotype index.

    Attributes
    ----------
    name : str
        The name of the phenotype.
    category : str
        The category of the phenotype.
    """

    name: str
    category: str


@dataclass
class PhenotypeInput:
    """Class that represents a phenotype input.

    Attributes
    ----------
    path : pathlib.Path | str
        The path to the phenotype data.
    pindex : PhenotypeIndex
        Phenotype index containing the name and category of the phenotype.
    index_format_func : Callable[[str], str]
        Function to format the index of the feature data.
        Eg: lambda x: x.strip().split("?")[-1].removesuffix(".RAST").removesuffix(".fna")
    """

    path: pathlib.Path | str
    pindex: PhenotypeIndex
    index_format_func: Callable[[str], str]


class Phenotype:
    """
    Class that represents a phenotype.

    Parameters
    ---------
    raw_phenotype_data : pd.Series
        Pandas Series containing the raw phenotype data.
    pindex : PhenotypeIndex
        Phenotype index containing the name and category of the phenotype.

    Attributes
    ---------
    phenotype_data : pd.Series
        Pandas Series containing the filtered phenotype data.
    pindex : PhenotypeIndex
        Phenotype index containing the name and category of the phenotype.
    """

    def __init__(self, raw_phenotype_data: pd.Series, pindex: PhenotypeIndex) -> None:
        self.pindex = pindex
        self._phenotype_data = self._parse_phenotype_data(raw_phenotype_data)

    def _parse_phenotype_data(self, raw_phenotype_data: pd.Series) -> pd.Series:
        """
        Parses the given raw phenotype data.

        Parameters
        ---------
        raw_phenotype_data : pd.Series
            Pandas Series containing the raw phenotype data.

        Returns
        ------
        pd.Series
            Pandas Series containing the filtered phenotype data.
        """
        undup_raw_phenotype_data = raw_phenotype_data.dropna().astype("uint8")
        return undup_raw_phenotype_data.loc[
            ~undup_raw_phenotype_data.index.duplicated(keep="first")
        ]

    def __repr__(self) -> str:
        size = self._phenotype_data.shape[0]
        return f"Phenotype (name={self.pindex.name}, category={self.pindex.category}, size={size})"

    def __hash__(self) -> int:
        return hash(self.pindex)

    @property
    def phenotype_data(self) -> pd.Series:
        """Pandas Series containing the filtered phenotype data."""
        return self._phenotype_data.copy(deep=True)

    @classmethod
    def read_data(cls, pinput: PhenotypeInput) -> "Phenotype":
        """Read the phenotype data from the file.

        Parameters
        ----------
        pinput : PhenotypeInput
            Phenotype input containing the path, PhenotypeIndex and index_format_func.

        Returns
        -------
        Phenotype
            The Phenotype object.
        """
        # NOTE: We use Int64 to handle NaN values
        file_path = pinput.path
        index_format_func = pinput.index_format_func
        raw_phenotype_df = pd.read_csv(
            file_path, sep="\t", index_col=0, dtype={"genomeID": str}
        ).astype("Int64")
        if raw_phenotype_df.index.name != "genomeID":
            raise ValueError("The index of the Phenotype table must be 'genomeID'")
        if raw_phenotype_df.shape[1] > 1:
            raise ValueError("The Phenotype table can only contain one phenotype")
        phenotype_data = raw_phenotype_df.iloc[:, 0]
        phenotype_data.index = phenotype_data.index.apply(index_format_func)
        return Phenotype(phenotype_data, pinput.pindex)

    def save(self, file_path: str | pathlib.Path) -> None:
        """
        Saves the phenotype data to the given path.

        Parameters
        ---------
        file_path : str | pathlib.Path
            The file path to the pickle file along with the extension
        """
        data = {
            "pindex": self.pindex,
            "_phenotype_data": self._phenotype_data,
        }
        with open(file_path, "wb") as fid:
            pickle.dump(data, fid)

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "Phenotype":
        """
        Loads the phenotype data from the given path.

        Parameters
        ---------
        file_path : str | pathlib.Path
            The file path to the pickle file along with the extension

        Returns
        ------
        Phenotype
            Phenotype object
        """
        with open(file_path, "rb") as fid:
            data = pickle.load(fid)
        phenotype = cls(data["_phenotype_data"], data["pindex"])
        return phenotype
