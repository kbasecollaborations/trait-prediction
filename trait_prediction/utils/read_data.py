"""Module that contains functions for reading phenotype data"""

import pathlib

import pandas as pd

from ..main import Phenotype


def read_phenotype_data(phenotype_data_path: pathlib.Path) -> list[Phenotype]:
    """
    Reads the phenotype data from the given file and returns it as a pandas DataFrame.

    Parameters
    ---------
    phenotype_data_path : pathlib.Path
        Path to the file containing the phenotype data.

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the phenotype data.
    """
    raw_phenotype_df = pd.read_csv(phenotype_data_path, sep="\t", index_col=0)
    phenotype_df = raw_phenotype_df.select_dtypes(include="number")
    phenotypes = []
    for col in phenotype_df.columns:
        name, category = col.split("--")
        if not category:
            category = "unknown"
        phenotypes.append(Phenotype(phenotype_df[col], name, category))
    return phenotypes
