"""Module that defines the Phenotype class"""

import pandas as pd


class Phenotype:
    """
    Class that represents a phenotype.

        Parameters
        ---------
        raw_data : pd.Series
            Pandas Series containing the raw phenotype data.
        name : str
            Name of the phenotype.
        category : str
            Category of the phenotype.

        Attributes
        ---------
        data : pd.Series
            Pandas Series containing the filtered phenotype data.
        name : str
            Name of the phenotype.
        category : str
            Category of the phenotype.
    """

    def __init__(self, raw_data: pd.Series, name: str, category: str) -> None:
        self.name = name
        self.category = category
        self.data = self._parse_data(raw_data)

    def __repr__(self) -> str:
        return f"Phenotype(name={self.name}, category={self.category}, size={self.data.shape})"

    def _parse_data(self, raw_data: pd.Series) -> pd.Series:
        """
        Parses the given raw phenotype data.

        Parameters
        ---------
        raw_data : pd.Series
            Pandas Series containing the raw phenotype data.

        Returns
        ------
        pd.Series
            Pandas Series containing the filtered phenotype data.
        """
        return raw_data.dropna()
