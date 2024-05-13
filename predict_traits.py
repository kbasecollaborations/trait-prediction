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
