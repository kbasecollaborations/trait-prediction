import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from ..pipeline.config import Config


def plot_shap_summary(
    clf,
    feature_data: pd.DataFrame,
    config: Config,
    title: str,
    output_file: str,
) -> pd.Series:
    """
    Plot SHAP summary plot.

    Parameters
    ----------
    clf
        Classifier object
    feature_data : pd.DataFrame
        Feature data.
    title : str
        The title of the plot (phenotype name)
    output_file : str
        Output file path.
    """
    feature_labels = list(feature_data.columns)
    explainer = shap.Explainer(clf)
    shap_values = explainer(feature_data)
    shap_values.feature_names = feature_labels
    # Summarize the SHAP values to get the mean absolute value for each feature
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    # Create a pandas Series for easy plotting and manipulation, with feature names
    importance_df = pd.Series(shap_sum, index=feature_labels).sort_values(
        ascending=False
    )
    shap.summary_plot(shap_values, max_display=config.shap_max_display, show=False)
    plt.title(title)
    shap_summary_plot = plt.gcf()
    shap_summary_plot.savefig(output_file)
    plt.clf()
    return importance_df
