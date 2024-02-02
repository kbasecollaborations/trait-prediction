"""Module that contains functions for reading data from files."""

import pathlib

import pandas as pd


def read_generic_features(
    generic_feature_file: pathlib.Path,
    bool_conversion: bool = False,
    dtype: str | None = None,
) -> pd.DataFrame:
    """
    Reads the generic features from the given file and returns them as a pandas DataFrame.

    Parameters
    ---------
    generic_feature_file : pathlib.Path
        Path to the file containing the generic features.
    bool_conversion : bool
        Flag indicating whether to convert the features to boolean
        Default is False
    dtype : str, optional
        Data type to use for the features. If None, the data type is inferred.
        Default is None

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the generic features.
    """
    if dtype is not None:
        generic_df = (
            pd.read_csv(
                generic_feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
            )
            .fillna(0)
            .astype(dtype)  # type: ignore
        )
    else:
        generic_df = pd.read_csv(
            generic_feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
        ).fillna(0)
    if generic_df.index.name != "genomeID":
        raise ValueError("The index of the Feature table must be 'genomeID'")
    if bool_conversion:
        generic_df[generic_df > 0] = 1
    return generic_df
