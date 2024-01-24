#!/usr/bin/env python3

import argparse
import gc
import json
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
from trait_prediction.utils.read_features import read_generic_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95
TEST_SIZE = 0.3
N_SPLITS = 5
PHENOTYPE_SAMPLE_SIZE_THRESHOLD = 10
MINOR_CLASS_SAMPLE_SIZE_THRESHOLD = 5
SHAP_MAX_DISPLAY = 10


def read_feature_data(
    feature_file: pathlib.Path, feature_type: str
) -> tuple[pd.DataFrame, dict]:
    """
    Read the feature data.

    Parameters
    ---------
    feature_file : pathlib.Path
        The path to the feature file.
    feature_type : str
        The type of the feature file.

    Returns
    ------
    tuple[pd.DataFrame, dict]
        Tuple of feature data and id dictionary.
    """
    if feature_type == "generic":
        features = read_generic_features(feature_file, bool_conversion=True)
        id_dict = {v: v for v in features.columns}
    elif feature_type == "rast":
        features, id_dict = read_rast_features(feature_file)
    elif feature_type == "interpro":
        features, id_dict = read_interpro_features(feature_file)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    return features, id_dict


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
    class_counts = phenotype.phenotype_data.value_counts()
    scores = {
        "name": phenotype.name,
        "category": phenotype.category,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "class_0": class_counts[0],
        "class_1": class_counts[1],
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
    cv_scores, estimators = phenotype_predictor.cross_validate_kfold(n_splits=n_splits)
    return cv_scores, estimators


def plot_shap_summary(
    predictor: PhenotypePredictor,
    feature_data: pd.DataFrame,
    id_dict: dict,
    output_file: str,
) -> None:
    """
    Plot SHAP summary plot.

    Parameters
    ----------
    predictor : PhenotypePredictor
        PhenotypePredictor object.
    feature_data : pd.DataFrame
        Feature data.
    id_dict : dict
        Maps annotation ids to annotation names for feature data.
    output_file : str
        Output file path.
    """
    clf = predictor.classifier
    feature_labels = [id_dict[c] for c in feature_data.columns]
    explainer = shap.Explainer(clf)
    shap_values = explainer(feature_data)
    shap_values.feature_names = feature_labels
    shap.summary_plot(shap_values, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"{predictor.phenotype.name}")
    shap_summary_plot = plt.gcf()
    shap_summary_plot.savefig(output_file)
    plt.clf()


def main(
    phenotype_file: pathlib.Path,
    feature_file: pathlib.Path,
    feature_type: str,
    random_state: int,
    results_folder: pathlib.Path,
    limit: int | None,
    score_func: str,
    n_features: int,
    cross_validate: bool,
    overwrite: bool,
    save_misc: bool,
) -> None:
    if limit is not None:
        phenotypeset = PhenotypeSet.limit(PhenotypeSet.read_data(phenotype_file), limit)
    else:
        phenotypeset = PhenotypeSet.read_data(phenotype_file)
    features, id_dict = read_feature_data(feature_file, feature_type)
    pbar = tqdm(phenotypeset)
    for phenotype in pbar:
        pbar.set_description(f"Processing {phenotype}")
        output_folder = results_folder / f"{phenotype.category}/{phenotype.name}"
        if (not overwrite) and output_folder.is_dir() and any(output_folder.iterdir()):
            pbar.set_description(f"Skipping {phenotype}")
            continue
        elif overwrite and output_folder.is_dir() and any(output_folder.iterdir()):
            for file in output_folder.iterdir():
                file.unlink()
            pbar.set_description(f"Overwriting {phenotype} results")
        else:
            output_folder.mkdir(parents=True, exist_ok=True)

        # process phenotype and feature data
        phenotype.set_feature_data(features, feature_type=feature_type)
        # TODO: Add feature_selection_kbest here
        (
            low_var_features,
            correlated_features_dict,
            low_score_features,
        ) = phenotype.filter_feature_data(
            variance_threshold=VARIANCE_THRESHOLD,
            correlation_treshold=CORRELATION_THRESHOLD,
            score_func=score_func,
            n_features=n_features,
        )
        # Skip if phenotype has less than 10 (default) samples
        if phenotype.phenotype_data.shape[0] <= PHENOTYPE_SAMPLE_SIZE_THRESHOLD:
            pbar.set_description(f"Skipping {phenotype}")
            continue
        # Skip if minor class of phenotype has less than 5 (default) samples
        if (
            phenotype.phenotype_data.value_counts().min()
            <= MINOR_CLASS_SAMPLE_SIZE_THRESHOLD
        ):
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
            test_size=TEST_SIZE, stratify=True, imbalanced="auto"
        )
        phenotype_predictor.fit()
        y_pred = phenotype_predictor.predict()
        y_true = data["y_test"]
        scores = get_scores(y_true, y_pred, phenotype)  # type: ignore

        # cross validation
        if cross_validate:
            cv_scores, _ = perform_cv(phenotype_predictor, n_splits=5)
        else:
            cv_scores = None

        # file writing
        scores_file = output_folder / "scores.csv"
        scores.to_csv(scores_file, index=True, sep=",")
        if cv_scores is not None:
            cv_scores_file = output_folder / "cv_scores.csv"
            cv_scores.to_csv(cv_scores_file, index=False, sep=",")

        # Save misc. files
        phenotype_predictor.save(output_folder)
        if save_misc:
            # save low var and high corr features to files
            with open(output_folder / "low_var_features.txt", "w") as fid:
                for low_var_feature in low_var_features:
                    fid.write(low_var_feature)
                    fid.write("\n")
            with open(output_folder / "corr_features.json", "w") as fid:
                json.dump(correlated_features_dict, fid)
            with open(output_folder / "low_score_features.txt", "w") as fid:
                for low_score_feature in low_score_features:
                    fid.write(low_score_feature)
                    fid.write("\n")

        # save shap summary plot
        shap_summary_plot_file = str(output_folder / "shap_summary_plot.png")
        plot_shap_summary(
            phenotype_predictor, phenotype.feature_data, id_dict, shap_summary_plot_file
        )

        # unset the feature data to reduce memory
        phenotype.unset_feature_data()

        # garbage collection
        del low_var_features
        del correlated_features_dict
        del low_score_features
        gc.collect()


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
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit the number of phenotypes"
    )
    parser.add_argument(
        "--score_func",
        type=str,
        default=None,
        help="Score function for feature selection",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=1000,
        help="Limit the number of features based on score_func",
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Perform cross validation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    parser.add_argument(
        "--save_misc",
        action="store_true",
        help="Save the phenotype and predictor as a pickle file",
    )
    args = parser.parse_args()
    phenotype_file = pathlib.Path(args.phenotype_file)
    feature_file = pathlib.Path(args.feature_file)
    feature_type = args.feature_type
    random_state = int(args.random_state)
    results_folder = pathlib.Path(args.results_folder)
    limit = args.limit
    score_func = args.score_func
    n_features = args.n_features
    cross_validate = args.cross_validate
    overwrite = args.overwrite
    save_misc = args.save_misc
    if phenotype_file.is_file() and feature_file.is_file():
        main(
            phenotype_file,
            feature_file,
            feature_type,
            random_state,
            results_folder,
            limit,
            score_func,
            n_features,
            cross_validate,
            overwrite,
            save_misc,
        )
    else:
        raise FileNotFoundError("Phenotype file or feature file not found")
