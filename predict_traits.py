#!/usr/bin/env python3

import argparse
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
    feature_dimensionality_reduction,
)
from trait_prediction.main import Phenotype, PhenotypeSet
from trait_prediction.training import PhenotypePredictor
from trait_prediction.utils.read_features import read_generic_features

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# NOTE: For binary feature, v = p(1-p), keep between (0.01-0.05)
VARIANCE_THRESHOLD = 0.01  # if v=0.01, p=0.01 or 0.99
CORRELATION_THRESHOLD = None  # NOTE: Disabling correlation filtering for now
IMBALANCED = None  # NOTE: Disabling imbalanced sampling for now
TEST_SIZE = 0.3
N_SPLITS = 5
PHENOTYPE_SAMPLE_SIZE_THRESHOLD = 20
MINOR_CLASS_SAMPLE_SIZE_THRESHOLD = 10
SHAP_MAX_DISPLAY = 10
SCORING = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "matthews_corrcoef",
)

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
    "NMF",
    "PCA",
]


def read_feature_data(
    feature_file: pathlib.Path,
    feature_type: str,
    bool_conversion: bool = True,
    dtype: str | None = "int64",
) -> pd.DataFrame:
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
        bool_conversion = False
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    features = read_generic_features(
        feature_file, bool_conversion=bool_conversion, dtype=dtype
    )
    return features


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
        thread_count=1,
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


def perform_cv(
    phenotype_predictor: PhenotypePredictor, n_splits: int, scoring: tuple
) -> tuple[pd.DataFrame | None, list | None, dict | None]:
    """
    Performs cross validation.

    Parameters
    ---------
    phenotype_predictor : PhenotypePredictor
        PhenotypePredictor object.
    n_splits : int
        Number of splits.
    scoring : tuple[str]
        The scoring metrics to use during cross validation
    """
    sampling_type = phenotype_predictor._sampling_params["sampling_type"]
    if sampling_type == "oversample":
        return None, None, None
    cv_scores, estimators, indices = phenotype_predictor.cross_validate_kfold(
        n_splits=n_splits, n_jobs=1, scoring=scoring
    )
    return cv_scores, estimators, indices


def plot_shap_summary(
    clf,
    feature_data: pd.DataFrame,
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
    shap.summary_plot(shap_values, max_display=SHAP_MAX_DISPLAY, show=False)
    plt.title(title)
    shap_summary_plot = plt.gcf()
    shap_summary_plot.savefig(output_file)
    plt.clf()
    return importance_df


def save_data(
    phenotype_fd: pd.DataFrame,
    phenotype_pd: pd.Series,
    phenotype_predictor: PhenotypePredictor,
    data: dict,
    low_var_features: list[str],
    correlated_features_dict: dict,
    low_score_features: list[str],
    scores: pd.DataFrame,
    cv_score_df: pd.DataFrame | None,
    cv_estimators: list | None,
    cv_indices: dict | None,
    output_folder: pathlib.Path,
    save_all: bool,
):
    # Step1: Save the pre-processing and feature data
    with open(output_folder / "low_var_features_list.txt", "w") as fid:
        fid.write("\n".join(low_var_features))
    with gzip.open(output_folder / "corr_features_map.json.gz", "wt") as gzfile:
        json.dump(correlated_features_dict, gzfile)
    with open(output_folder / "low_score_features_list.txt", "w") as fid:
        fid.write("\n".join(low_score_features))
    with open(output_folder / "features_list.txt", "w") as fid:
        fid.write("\n".join(phenotype_fd.columns))
    for key in ["y_train", "y_test"]:
        data[key].to_csv(output_folder / f"{key}.tsv", index=True, sep="\t")

    # Step 2a: Save the scores
    scores_file = output_folder / "scores.csv"
    scores.to_csv(scores_file, index=True, sep=",")
    # Step2b: Save CV results
    phenotype_index = phenotype_pd.index
    if cv_score_df is not None and cv_indices is not None:
        cv_score_df["name"] = phenotype_predictor.phenotype.name
        cv_score_df["category"] = phenotype_predictor.phenotype.category
        cv_scores_file = output_folder / "cv_scores.csv"
        cv_score_df.to_csv(cv_scores_file, index=True, sep=",")
        train_genomes = pd.DataFrame(
            [phenotype_index[i] for i in cv_indices["train"]], dtype=str
        )
        train_genomes.to_csv(
            output_folder / "cv_train_genomes.csv", index=True, sep=","
        )
        val_genomes = pd.DataFrame(
            [phenotype_index[i] for i in cv_indices["test"]], dtype=str
        )
        val_genomes.to_csv(output_folder / "cv_val_genomes.csv", index=True, sep=",")
    # Step 2c: Save model and phenotype data
    if save_all:
        phenotype_predictor.classifier.save_model(output_folder / "model.cbm")
        phenotype_predictor.save(output_folder)
        if cv_estimators is not None:
            for i, model in enumerate(cv_estimators):
                model.save_model(output_folder / f"cv_model_{i}.cbm")

    # Step 3a: Save SHAP summary plot and top features
    shap_summary_plot_file = str(output_folder / "shap_summary_plot.png")
    importance_df = plot_shap_summary(
        phenotype_predictor.classifier,
        data["X_train"],
        title=phenotype_predictor.phenotype.name,
        output_file=shap_summary_plot_file,
    )
    importance_df.to_csv(output_folder / "shap_features.csv", index=True, sep=",")
    # Step 3b: Save SHAP summary plot and top features for CV
    if cv_indices is not None and cv_estimators is not None:
        for i, model in enumerate(cv_estimators):
            genomes = phenotype_index[cv_indices["train"][i]]
            shap_summary_plot_file = str(
                output_folder / f"cv_shap_summary_plot_{i}.png"
            )
            importance_df = plot_shap_summary(
                model,
                phenotype_fd.loc[genomes, :],
                title=phenotype_predictor.phenotype.name,
                output_file=shap_summary_plot_file,
            )
            importance_df.to_csv(
                output_folder / f"cv_shap_features_{i}.csv", index=True, sep=","
            )


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
    phenotype.set_feature_data(features, feature_type=feature_type)
    if features.shape[1] <= 30_000:
        corr_method = "numpy"
    else:
        corr_method = "numba"
    if feature_type in FLOAT_FEATURES:
        var_thres = 0.0
    else:
        var_thres = VARIANCE_THRESHOLD
    (
        low_var_features,
        correlated_features_dict,
        low_score_features,
    ) = phenotype.filter_feature_data(
        variance_threshold=var_thres,
        correlation_treshold=CORRELATION_THRESHOLD,
        score_func=score_func,
        n_features=n_features,
        method=corr_method,
    )
    # Check data
    phenotype_pd = phenotype.phenotype_data
    phenotype_fd = phenotype.feature_data
    if not is_data_good(phenotype_pd):
        return None

    # Step 4: Make classifier and predict
    cross_validate = params["cross_validate"]
    if feature_type in FLOAT_FEATURES or reduction_func is not None:
        categorical_feature_names = None
    else:
        categorical_feature_names = []
        for col in phenotype_fd.columns:
            col_dtype = str(phenotype_fd[col].dtype)
            if col_dtype.startswith("uint"):
                categorical_feature_names.append(col)
    clf = make_classifier(random_state, categorical_feature_names)
    phenotype_predictor = PhenotypePredictor(phenotype, clf, random_state=random_state)
    # TODO: Enable imbalanced sampling again
    data = phenotype_predictor.split_data(
        test_size=TEST_SIZE, stratify=True, imbalanced=IMBALANCED
    )
    phenotype_predictor.fit()
    y_pred = phenotype_predictor.predict()
    y_true = data["y_test"]
    scores = get_scores(y_true, y_pred, phenotype)  # type: ignore

    # Cross validation
    if cross_validate:
        clf_cv = make_classifier(random_state, categorical_feature_names)
        phenotype_predictor_cv = PhenotypePredictor(
            phenotype, clf_cv, random_state=random_state
        )
        data_cv = phenotype_predictor_cv.split_data(
            test_size=0.0, stratify=True, imbalanced=IMBALANCED
        )
        cv_score_df, cv_estimators, cv_indices = perform_cv(
            phenotype_predictor_cv, n_splits=5, scoring=SCORING
        )
    else:
        cv_score_df, cv_estimators, cv_indices = None, None, None

    # TODO: Validation
    # scores = get_scores(y_true, y_pred, phenotype)  # type: ignore

    # Step 5: File writing
    save_all = params["save_all"]
    save_data(
        phenotype_fd,
        phenotype_pd,
        phenotype_predictor,
        data,
        low_var_features,
        correlated_features_dict,
        low_score_features,
        scores,
        cv_score_df,
        cv_estimators,
        cv_indices,
        output_folder,
        save_all,
    )

    # Unset the feature data to reduce memory usage
    phenotype.unset_feature_data()


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
    n_cpus: int,
    save_all: bool,
    overwrite: bool,
) -> None:
    if limit is not None:
        phenotypeset = PhenotypeSet.limit(PhenotypeSet.read_data(phenotype_file), limit)
    else:
        phenotypeset = PhenotypeSet.read_data(phenotype_file)
    features = read_feature_data(feature_file, feature_type)
    if reduction_func is not None:
        features_reduc_file = results_folder / f"features_{reduction_func}.tsv"
        components_file = results_folder / "components.tsv"
        if features_reduc_file.is_file() and components_file.is_file():
            features = read_feature_data(features_reduc_file, reduction_func)
        else:
            features, components_df = feature_dimensionality_reduction(
                features,
                method=reduction_func,
                n_components=n_features,
                random_state=random_state,
            )
            features.to_csv(features_reduc_file, sep="\t", index=True)
            components_df.to_csv(components_file, index=True, sep="\t")
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
            "save_all": save_all,
            "features": features,
            "phenotype": phenotype,
        }
        mp_args.append(mp_arg)
    with mp.Pool(processes=n_cpus) as p:
        results = []
        for result in tqdm(p.imap(train_model, mp_args), total=len(mp_args)):
            results.append(result)


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
        default="None",
        help="Score function for feature selection",
    )
    parser.add_argument(
        "--reduction_func",
        type=str,
        default="None",
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
        "--n_cpus",
        type=int,
        default=-1,
        help="Number of processes to use",
    )
    parser.add_argument(
        "--save_all",
        action="store_true",
        help="Save model and phenotype pkl files",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results",
    )
    # TODO: Add argument to enable/disable imbalanced sampling
    args = parser.parse_args()
    phenotype_file = pathlib.Path(args.phenotype_file)
    feature_file = pathlib.Path(args.feature_file)
    feature_type = args.feature_type
    random_state = args.random_state
    results_folder = pathlib.Path(args.results_folder)
    limit = args.limit
    score_func = args.score_func if args.score_func != "None" else None
    reduction_func = args.reduction_func if args.reduction_func != "None" else None
    if score_func is not None and reduction_func is not None:
        raise ValueError("Both score_func and reduction_func cannot be set")
    n_features = args.n_features
    cross_validate = args.cross_validate
    n_cpus = args.n_cpus if args.n_cpus > 0 else mp.cpu_count()
    save_all = args.save_all
    overwrite = args.overwrite
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
            n_cpus,
            save_all,
            overwrite,
        )
    else:
        raise FileNotFoundError("Phenotype file or feature file not found")
