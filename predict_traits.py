#!/usr/bin/env python3

import argparse
import multiprocessing as mp
import pathlib
import warnings

import pandas as pd
from catboost import CatBoostClassifier
from tqdm import tqdm

from trait_prediction.feature_selection.reduction import (
    feature_dimensionality_reduction,
)
from trait_prediction.main import Phenotype, PhenotypeSet
from trait_prediction.training import PhenotypePredictor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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
        # iterations=1000,
        # depth=8,
        # learning_rate=0.03,
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
    features_map = params["features_map"]
    feature_type = params["feature_type"]
    score_func = params["score_func"]
    reduction_func = params["reduction_func"]
    n_features = params["n_features"]
    random_state = params["random_state"]
    if features_map is not None:
        features = features_map[phenotype.name]
        print(f"Using features combined features for {phenotype.name}")
    if features is None and features_map is None:
        raise ValueError("Both features and features_map cannot be None")
    if features.shape[0] == 0 or features.shape[1] == 0:
        print(f"No features found for {phenotype.name}")
        return None
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
        _ = phenotype_predictor_cv.split_data(
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
    feature_file: pathlib.Path | list[pathlib.Path],
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
    if isinstance(feature_file, list):
        features_map = {
            f.stem.removesuffix("_features_combined"): read_feature_data(
                f, feature_type
            )
            for f in feature_file
        }
        features = None
    else:
        features = read_feature_data(feature_file, feature_type)
        features_map = None
    if reduction_func is not None and features is not None:
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
            "features_map": features_map,
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
    elif (
        phenotype_file.is_file()
        and feature_file.is_dir()
        and feature_type == "combined"
    ):
        feature_file_list = list(feature_file.glob("*.tsv"))
        main(
            phenotype_file,
            feature_file_list,
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
