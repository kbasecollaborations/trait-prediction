#!/usr/bin/env python3

import argparse
import gc
import gzip
import json
import multiprocessing as mp
import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
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

from trait_prediction.feature_selection.reduction import (
    remove_features_with_high_correlation,
    remove_features_with_low_variance,
)
from trait_prediction.main import Phenotype, PhenotypeSet
from trait_prediction.training import PhenotypePredictor
from trait_prediction.utils.read_features import read_generic_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# For binary feature, v = p(1-p), keep between (0.01-0.05)
VARIANCE_THRESHOLD = 0.05  # if v=0.01, p=0.01 or 0.99
CORRELATION_THRESHOLD = 0.95
TEST_SIZE = 0.3
N_SPLITS = 5
PHENOTYPE_SAMPLE_SIZE_THRESHOLD = 20
MINOR_CLASS_SAMPLE_SIZE_THRESHOLD = 10
SHAP_MAX_DISPLAY = 10

COUNT_FEATURES = [
    "rast",
    "kofam",
    "uniref30",
    "cluster30",
    "eggnog_kegg",
    "uniprot_trembl",
    "cluster70",
    "uniref90",
    "eggnog_seed",
    "cluster50",
    "cluster90",
]
FLOAT_FEATURES = [
    "kofam_modules",
]


def read_feature_data(
    feature_file: pathlib.Path,
    feature_type: str,
    bool_conversion: bool = True,
    dtype: str | None = "int64",
) -> tuple[pd.DataFrame, dict]:
    """
    Read the feature data.

    Parameters
    ---------
    feature_file : pathlib.Path
        The path to the feature file.
    feature_type : str
        The type of the feature file.
    bool_conversion : bool
        Flag indicating whether to convert the features to boolean.
        Default is True.

    Returns
    ------
    tuple[pd.DataFrame, dict]
        Tuple of feature data and id dictionary.
    """
    if feature_type in COUNT_FEATURES:
        if bool_conversion:
            dtype = "uint8"
        else:
            dtype = "uint32"
    elif feature_type in FLOAT_FEATURES:
        dtype = "float64"
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    features = read_generic_features(
        feature_file, bool_conversion=bool_conversion, dtype=dtype
    )
    id_dict = {v: v for v in features.columns}
    return features, id_dict


def is_data_good(phenotype_data: pd.Series) -> bool:
    """
    Check if the phenotype data is good for training.

    Parameters
    ----------
    phenotype_data : pd.Series
        Phenotype data.

    Returns
    -------
    bool
        True if the data is good for training, otherwise False.
    """
    # Skip if phenotype has less than 10 (default) samples
    if phenotype_data.shape[0] <= PHENOTYPE_SAMPLE_SIZE_THRESHOLD:
        return False
    # Skip if minor class of phenotype has less than 5 (default) samples
    if phenotype_data.value_counts().min() <= MINOR_CLASS_SAMPLE_SIZE_THRESHOLD:
        return False
    # Skip if phenotype has only one class
    if len(phenotype_data.unique()) == 1:
        return False
    return True


def make_classifier(random_state: int, categorical_feature_names: list[str] | None):
    """
    Creates a classifier.

    Parameters
    ---------
    random_state : int
        Random state.
    categorical_feature_names : list[str] | None
        List of categorical feature names.

    Returns
    ------
    Classifier object.
    """
    clf = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.03,
        # l2_leaf_reg=3,
        # bootstrap_type="Bayesian",
        # bagging_temperature=1,
        random_state=random_state,
        objective="Logloss",
        cat_features=categorical_feature_names,
        verbose=False,
        allow_writing_files=False,
    )
    return clf


def get_scores(
    y_true: pd.Series, y_pred: pd.Series, phenotype: Phenotype
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
) -> pd.Series:
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
    # Summarize the SHAP values to get the mean absolute value for each feature
    shap_sum = np.abs(shap_values.values).mean(axis=0)
    # Create a pandas Series for easy plotting and manipulation, with feature names
    importance_df = pd.Series(shap_sum, index=feature_labels).sort_values(
        ascending=False
    )
    shap.summary_plot(shap_values, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(f"{predictor.phenotype.name}")
    shap_summary_plot = plt.gcf()
    shap_summary_plot.savefig(output_file)
    plt.clf()
    return importance_df


def save_data(
    phenotype_fd: pd.DataFrame,
    components_df: pd.DataFrame | None,
    data: dict,
    phenotype_predictor: PhenotypePredictor,
    output_folder: pathlib.Path,
    id_dict: dict,
    low_var_features: list[str],
    correlated_features_dict: dict,
    low_score_features: list[str],
    scores: pd.DataFrame,
    cv_scores: pd.DataFrame | None,
):
    # Step1: Save the feature and training data
    if components_df is not None:
        with gzip.open(output_folder / "components.tsv.gz", "wt") as gzfile:
            components_df.to_csv(gzfile, index=True, sep="\t")
    with open(output_folder / "low_var_features.txt", "w") as fid:
        fid.write("\n".join(low_var_features))
    with gzip.open(output_folder / "corr_features.json.gz", "wt") as gzfile:
        gzfile.write(json.dumps(correlated_features_dict).encode("utf-8"))  # type: ignore
    with open(output_folder / "low_score_features.txt", "w") as fid:
        fid.write("\n".join(low_score_features))
    for key in ["X_train", "X_test", "y_train", "y_test"]:
        data[key].to_csv(output_folder / f"{key}.tsv", index=True, sep="\t")

    # Step 2: Save the scores
    scores_file = output_folder / "scores.csv"
    scores.to_csv(scores_file, index=True, sep=",")
    if cv_scores is not None:
        cv_scores_file = output_folder / "cv_scores.csv"
        cv_scores.to_csv(cv_scores_file, index=False, sep=",")

    # phenotype_predictor.save(output_folder)

    # Step 3: Save SHAP summary plot and top features
    shap_summary_plot_file = str(output_folder / "shap_summary_plot.png")
    importance_df = plot_shap_summary(
        phenotype_predictor, phenotype_fd, id_dict, shap_summary_plot_file
    )
    importance_df.to_csv(output_folder / "shap_features.csv", index=True, sep=",")


def train_model(params: dict) -> None:
    """
    Train model to predict phenotype using gene annotation feature data

    Parameters
    ---------
    params : dict
        Params for model training
    """
    # Step 1: Processing the phenotype
    results_folder = params["results_folder"]
    phenotype = params["phenotype"]
    overwrite = params["overwrite"]
    output_folder = results_folder / f"{phenotype.category}/{phenotype.name}"
    if (not overwrite) and output_folder.is_dir() and any(output_folder.iterdir()):
        return None
    elif overwrite and output_folder.is_dir() and any(output_folder.iterdir()):
        for file in output_folder.iterdir():
            file.unlink()
    else:
        output_folder.mkdir(parents=True, exist_ok=True)

    # Step 2: Filtering the phenotype
    features = params["features"]
    feature_type = params["feature_type"]
    score_func = params["score_func"]
    reduction_func = params["reduction_func"]
    n_features = params["n_features"]
    random_state = params["random_state"]
    phenotype_pd = phenotype.phenotype_data
    phenotype_fd = phenotype.feature_data
    phenotype.set_feature_data(features, feature_type=feature_type)
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
    if reduction_func is not None:
        components_df = phenotype.reduce_feature_data(
            reduction_func, n_components=n_features, random_state=random_state
        )
    else:
        components_df = None
    # Check data
    if not is_data_good(phenotype_pd):
        return None

    # Step 4: Make classifier and predict
    cross_validate = params["cross_validate"]
    if feature_type in FLOAT_FEATURES:
        categorical_feature_names = None
    else:
        categorical_feature_names = []
        for col in phenotype_fd.columns:
            col_dtype = str(phenotype_fd[col].dtype)
            if col_dtype.startswith("uint"):
                categorical_feature_names.append(col)
    clf = make_classifier(random_state, categorical_feature_names)
    phenotype_predictor = PhenotypePredictor(phenotype, clf, random_state=random_state)
    data = phenotype_predictor.split_data(
        test_size=TEST_SIZE, stratify=True, imbalanced="auto"
    )
    phenotype_predictor.fit()
    y_pred = phenotype_predictor.predict()
    y_true = data["y_test"]
    scores = get_scores(y_true, y_pred, phenotype)  # type: ignore
    # Cross validation
    if cross_validate:
        cv_scores, _ = perform_cv(phenotype_predictor, n_splits=5)
    else:
        cv_scores = None

    # Step 5: File writing
    id_dict = params["id_dict"]
    # FIXME: Change order of arguments and number
    save_data(
        phenotype_fd,
        components_df,
        data,
        phenotype_predictor,
        output_folder,
        id_dict,
        low_var_features,
        correlated_features_dict,
        low_score_features,
        scores,
        cv_scores,
    )

    # Unset the feature data to reduce memory usage
    phenotype.unset_feature_data()
    # Garbage collection
    del low_var_features
    del correlated_features_dict
    del low_score_features
    gc.collect()


def main(
    phenotype_file: pathlib.Path,
    feature_file: pathlib.Path,
    feature_type: str,
    random_state: int,
    results_folder: pathlib.Path,
    limit: int | None,
    score_func: str | None,
    reduction_func: str | None,
    n_features: int,
    cross_validate: bool,
    overwrite: bool,
    n_cpus: int,
) -> None:
    if limit is not None:
        phenotypeset = PhenotypeSet.limit(PhenotypeSet.read_data(phenotype_file), limit)
    else:
        phenotypeset = PhenotypeSet.read_data(phenotype_file)
    features, id_dict = read_feature_data(feature_file, feature_type)
    features, low_var_features = remove_features_with_low_variance(
        features, threshold=VARIANCE_THRESHOLD
    )
    features, correlated_features_dict = remove_features_with_high_correlation(
        features, threshold=CORRELATION_THRESHOLD
    )
    features.to_csv(results_folder / "features.tsv", sep="\t", index=True)
    with open(results_folder / "low_var_features.txt", "w") as fid:
        fid.write("\n".join(low_var_features))
    with gzip.open(results_folder / "corr_features.json.gz", "wt") as gzfile:
        gzfile.write(json.dumps(correlated_features_dict).encode("utf-8"))  # type: ignore
    mp_args = []
    for phenotype in phenotypeset:
        mp_arg = {
            "feature_type": feature_type,
            "random_state": random_state,
            "results_folder": results_folder,
            "score_func": score_func,
            "reduction_func": reduction_func,
            "n_features": n_features,
            "cross_validate": cross_validate,
            "overwrite": overwrite,
            "features": features,
            "phenotype": phenotype,
            "id_dict": id_dict,
        }
        mp_args.append(mp_arg)
    with mp.Pool(processes=n_cpus) as p:
        results = tqdm(p.imap(train_model, mp_args), total=len(mp_args))


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
        "--reduction_func",
        type=str,
        default=None,
        help="Reduction function for feature dimensionality reduction",
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
        "--n_cpus",
        type=int,
        default=-1,
        help="Number of processes to use",
    )
    args = parser.parse_args()
    phenotype_file = pathlib.Path(args.phenotype_file)
    feature_file = pathlib.Path(args.feature_file)
    feature_type = args.feature_type
    random_state = args.random_state
    results_folder = pathlib.Path(args.results_folder)
    limit = args.limit
    score_func = args.score_func if args.score_func is not "None" else None
    reduction_func = args.reduction_func if args.reduction_func is not "None" else None
    if score_func is not None and reduction_func is not None:
        raise ValueError("Both score_func and reduction_func cannot be set")
    n_features = args.n_features
    cross_validate = args.cross_validate
    overwrite = args.overwrite
    n_cpus = args.ncpus if args.n_cpus > 0 else mp.cpu_count()
    if phenotype_file.is_file() and feature_file.is_file():
        main(
            phenotype_file,
            feature_file,
            feature_type,
            random_state,
            results_folder,
            limit,
            score_func,
            reduction_func,
            n_features,
            cross_validate,
            overwrite,
            n_cpus,
        )
    else:
        raise FileNotFoundError("Phenotype file or feature file not found")
