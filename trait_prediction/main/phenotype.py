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

    @classmethod
    def read_data(cls, file_path: str | pathlib.Path, category: str) -> "Phenotype":
        """Read the phenotype data from the file.

        Parameters
        ----------
        file_path : str | pathlib.Path
            The file path to the phenotype data.
        category : str
            The category of the phenotype.

        Returns
        -------
        "Phenotype"
            The Phenotype object.
        """
        # NOTE: We use Int64 to handle NaN values
        raw_phenotype_df = pd.read_csv(
            file_path, sep="\t", index_col=0, dtype={"genomeID": str}
        ).astype("Int64")
        if raw_phenotype_df.index.name != "genomeID":
            raise ValueError("The index of the Phenotype table must be 'genomeID'")
        if raw_phenotype_df.shape[1] > 1:
            raise ValueError("The Phenotype table can only contain one phenotype")
        name = str(raw_phenotype_df.columns[0])
        phenotype_data = raw_phenotype_df.loc[:, name]
        return Phenotype(phenotype_data, name, category)

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
