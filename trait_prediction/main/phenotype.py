"""Module that defines the Phenotype class"""

import pathlib
import pickle

import pandas as pd


class Phenotype:
    """
    Class that represents a phenotype.

    Parameters
    ---------
    raw_phenotype_data : pd.Series
        Pandas Series containing the raw phenotype data.
    name : str
        Name of the phenotype.
    category : str
        Category of the phenotype.

    Attributes
    ---------
    phenotype_data : pd.Series
        Pandas Series containing the filtered phenotype data.
    name : str
        Name of the phenotype.
    category : str
        Category of the phenotype.
    """

    def __init__(self, raw_phenotype_data: pd.Series, name: str, category: str) -> None:
        self.name = name
        self.category = category
        self._phenotype_data = self._parse_phenotype_data(raw_phenotype_data)

    def __repr__(self) -> str:
        size = self._phenotype_data.shape[0]
        return f"Phenotype (name={self.name}, category={self.category}, size={size})"

    def __hash__(self) -> int:
        unique_id = {
            "name": self.name,
            "category": self.category,
        }
        return hash(unique_id)

    @property
    def phenotype_data(self) -> pd.Series:
        """Pandas Series containing the filtered phenotype data."""
        return self._phenotype_data.copy(deep=True)

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

    def save(self, file_path: str | pathlib.Path) -> None:
        """
        Saves the phenotype data to the given path.

        Parameters
        ---------
        file_path : str | pathlib.Path
            The file path to the pickle file along with the extension
        """
        data = {
            "name": self.name,
            "category": self.category,
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
        phenotype = cls(data["_phenotype_data"], data["name"], data["category"])
        return phenotype
