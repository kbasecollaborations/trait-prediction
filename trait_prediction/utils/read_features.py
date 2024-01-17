"""Module that contains functions for reading data from files."""

import pathlib

import pandas as pd


def read_generic_features(
    generic_feature_file: pathlib.Path, bool_conversion: bool = False
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

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the generic features.
    """
    converter = bool if bool_conversion else int
    generic_df = (
        pd.read_csv(generic_feature_file, sep="\t", index_col=0)
        .fillna(0)
        .astype(converter)
    )
    return generic_df


def read_rast_features(
    rast_feature_file: pathlib.Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Reads the rast features from the given file and returns them as a pandas DataFrame.

    Parameters
    ---------
    rast_feature_file : pathlib.Path
        Path to the file containing the rast features.

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the rast features.
    dict[str, str]
        Dictionary mapping the rast subsystem ontology (SSO) to the full name.
    """
    rast_df = (
        pd.read_csv(rast_feature_file, sep="\t", index_col=0).fillna(0).astype(int)
    )
    rast_annotations = list(rast_df.columns)
    rast_sso_dict = {x.split("__")[0]: x.split("__", 1)[-1] for x in rast_annotations}
    rast_sso = list(rast_sso_dict.keys())
    rast_df.columns = rast_sso
    return rast_df, rast_sso_dict


def read_interpro_features(
    interpro_feature_file: pathlib.Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Reads the interpro features from the given file and returns them as a pandas DataFrame.

    Parameters
    ---------
    interpro_feature_file : pathlib.Path
        Path to the file containing the interpro features.

    Returns
    ------
    pd.DataFrame
        Pandas DataFrame containing the interpro features.
    dict[str, str]
        Dictionary mapping the interpro ontology (IPR) to the full name.
    """
    interpro_df = (
        pd.read_csv(interpro_feature_file, sep="\t", index_col=0).fillna(0).astype(int)
    )
    interpro_annotations = list(interpro_df.columns)
    interpro_ipr_dict = {
        x.split("__")[0]: x.split("__", 1)[-1] for x in interpro_annotations
    }
    interpro_ipr = list(interpro_ipr_dict.keys())
    interpro_df.columns = interpro_ipr
    return interpro_df, interpro_ipr_dict
