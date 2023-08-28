"""Module that defines the Phenotype class"""

import pandas as pd

from ..feature_selection.reduction import (
    remove_features_with_high_correlation,
    remove_features_with_low_variance,
)


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
        feature_data : union[pd.DataFrame, None]
            Pandas DataFrame containing the feature data.
        feature_type : union[str, None]
            Type of the feature data.
    """

    def __init__(self, raw_phenotype_data: pd.Series, name: str, category: str) -> None:
        self.name = name
        self.category = category
        self.phenotype_data = self._parse_phenotype_data(raw_phenotype_data)
        self.feature_data = None
        self.feature_type = None

    def __repr__(self) -> str:
        return f"Phenotype(name={self.name}, category={self.category}, size={self.phenotype_data.shape})"

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
        return raw_phenotype_data.dropna()

    def set_feature_data(
        self, raw_feature_data: pd.DataFrame, feature_type: str
    ) -> None:
        """
        Sets the feature data for this phenotype.

        Parameters
        ---------
        raw_feature_data : pd.DataFrame
            Pandas DataFrame containing the feature data.
        feature_type : str
            Type of the feature data.
        """
        if self.feature_data is None:
            common_genomes = sorted(
                list(
                    set(self.phenotype_data.index).intersection(
                        set(raw_feature_data.index)
                    )
                )
            )
            self.feature_data = raw_feature_data.loc[common_genomes, :]
            self.feature_type = feature_type
        else:
            raise ValueError("Feature data already set for this phenotype")

    def filter_feature_data(
        self, variance_threshold: float = 0.05, correlation_treshold: float = 0.95
    ) -> tuple[list[str], dict[str, list[str]]]:
        """
        Filters the feature data for this phenotype.

        Parameters
        ---------
        variance_threshold : float
            Threshold for the variance of the features.
            Default value 0.05
        correlation_treshold : float
            Threshold for the correlation of the features.
            Default value 0.95

        Returns
        -------
        list[str]
            List of the features with low variance that were removed
        dict[str, list[str]]
            Dictionary of the features with high correlation that were removed

        """
        if self.feature_data is not None:
            fd_high_var, low_var_features = remove_features_with_low_variance(
                self.feature_data, variance_threshold
            )
            (
                fd_high_var_low_corr,
                corr_group_dict,
            ) = remove_features_with_high_correlation(fd_high_var, correlation_treshold)
        else:
            raise ValueError("Feature data not set for this phenotype")
        return low_var_features, corr_group_dict
