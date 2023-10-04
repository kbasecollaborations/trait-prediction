#!/usr/bin/env python3

import argparse
import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from trait_prediction.main import Phenotype, PhenotypeSet
from trait_prediction.training import PhenotypePredictor
from trait_prediction.utils import read_interpro_features, read_rast_features

warnings.filterwarnings("ignore", category=UserWarning)


def make_classifier(random_state: int, categorical_feature_names: list[str]):
    """
    Creates a classifier.

    Parameters
    ---------
    random_state : int
        Random state.

    Returns
    ------
    Classifier object.
    """
    clf = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.03,
        random_state=random_state,
        loss_function="Logloss",
        cat_features=categorical_feature_names,
        verbose=False,
        allow_writing_files=False,
    )
    return clf


def get_scores(
    y_true: pd.DataFrame, y_pred: pd.DataFrame, phenotype: Phenotype
) -> pd.DataFrame:
    """
    Calculates the scores for the given true and predicted labels.

    Parameters
    ---------
    y_true : pd.DataFrame
        True labels.
    y_pred : pd.DataFrame
        Predicted labels.

    Returns
    ------
    pd.DataFrame
        Scores.
    """
    scores = {
        "name": phenotype.name,
        "category": phenotype.category,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
    }
    index = f"{phenotype.name}-{phenotype.category}"
    return pd.DataFrame(scores, index=[index])


def perform_cv(phenotype_predictor: PhenotypePredictor, n_splits: int):
    """
    Performs cross validation.

    Parameters
    ---------
    phenotype_predictor : PhenotypePredictor
        PhenotypePredictor object.
    n_splits : int
        Number of splits.
    """
    sampling_type = phenotype_predictor._sampling_params["sampling_type"]
    if sampling_type == "oversample":
        return None, None
    else:
        cv_scores, estimators = phenotype_predictor.cross_validate_kfold(
            n_splits=n_splits
        )
        return cv_scores, estimators


def main(
    phenotype_file: pathlib.Path,
    feature_file: pathlib.Path,
    feature_type: str,
    random_state: int,
    results_folder: pathlib.Path,
) -> None:
    phenotypeset = PhenotypeSet.read_data(phenotype_file)
    if feature_type == "rast":
        features, id_dict = read_rast_features(feature_file)
    elif feature_type == "interpro":
        features, id_dict = read_interpro_features(feature_file)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    pbar = tqdm(phenotypeset)
    for phenotype in pbar:
        pbar.set_description(f"Processing {phenotype}")
        output_folder = results_folder / f"{phenotype.category}/{phenotype.name}"
        if output_folder.is_dir() and any(output_folder.iterdir()):
            pbar.set_description(f"Skipping {phenotype}")
            continue
        else:
            output_folder.mkdir(parents=True, exist_ok=True)

        # process phenotype and feature data
        phenotype.set_feature_data(features, feature_type=feature_type)
        low_var_features, correlated_features_dict = phenotype.filter_feature_data(
            variance_threshold=0.01, correlation_treshold=0.95
        )
        # Skip if phenotype has less than 10 samples
        if phenotype.phenotype_data.shape[0] <= 10:
            pbar.set_description(f"Skipping {phenotype}")
            continue
        # Skip if minor class of phenotype has less than 5 samples
        if phenotype.phenotype_data.value_counts().min() <= 5:
            pbar.set_description(f"Skipping {phenotype}")
            continue
        # Skip if phenotype has only one class
        if len(phenotype.phenotype_data.unique()) == 1:
            pbar.set_description(f"Skipping {phenotype}")
            continue

        # make classifier and predict
        categorical_feature_names = phenotype.feature_data.columns.to_list()
        clf = make_classifier(random_state, categorical_feature_names)
        phenotype_predictor = PhenotypePredictor(
            phenotype, clf, random_state=random_state
        )
        data = phenotype_predictor.split_data(
            test_size=0.3, stratify=True, imbalanced="auto"
        )
        phenotype_predictor.fit()
        y_pred = phenotype_predictor.predict()
        y_true = data["y_test"]
        scores = get_scores(y_true, y_pred, phenotype)  # type: ignore

        # cross validation
        cv_scores, _ = perform_cv(phenotype_predictor, n_splits=5)
        # shap values
        explainer = shap.Explainer(clf)
        shap_values = explainer(phenotype.feature_data)
        shap.summary_plot(shap_values, max_display=10, show=False)
        # FIXME: The plots needs to be clearled befoe plotting the next thing
        shap_summary_plot = plt.gcf()

        # file writing
        phenotype_predictor.save(output_folder)
        scores_file = output_folder / "scores.csv"
        scores.to_csv(scores_file, index=True, sep=",")
        if cv_scores is not None:
            cv_scores_file = output_folder / "cv_scores.csv"
            cv_scores.to_csv(cv_scores_file, index=False, sep=",")
        shap_summary_plot_file = str(output_folder / "shap_summary_plot.png")
        shap_summary_plot.savefig(shap_summary_plot_file)

        # unset the feature data to reduce memory
        phenotype.unset_feature_data()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phenotype_file", type=str, help="The path to the phenotype file"
    )
    parser.add_argument("feature_file", type=str, help="The path to the feature file")
    parser.add_argument(
        "results_folder", type=str, help="The folder to save the results"
    )
    parser.add_argument("--feature_type", type=str, help="The type of the feature file")
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
    args = parser.parse_args()
    phenotype_file = pathlib.Path(args.phenotype_file)
    feature_file = pathlib.Path(args.feature_file)
    feature_type = args.feature_type
    random_state = args.random_state
    results_folder = pathlib.Path(args.results_folder)
    if phenotype_file.is_file() and feature_file.is_file():
        main(phenotype_file, feature_file, feature_type, random_state, results_folder)
    else:
        raise FileNotFoundError("Phenotype file or feature file not found")
