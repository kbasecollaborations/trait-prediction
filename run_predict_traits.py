#!/usr/bin/env python

import argparse
import multiprocessing as mp
import pathlib
import subprocess

# DATASETS = ["ch", "pmi", "leaf"]
DATASETS = ["ch"]
FEATURES = [
    "rast",
    "kofam",
    "uniref30",
    "cluster30",
    "eggnog_kegg",
    "uniprot_trembl",
    "cluster70",
    "uniref90",
    "kofam_modules",
    "eggnog_seed",
    "cluster50",
    "cluster90",
]


def main(
    phenotypes_folder: pathlib.Path,
    features_folder: pathlib.Path,
    outputs_folder: pathlib.Path,
    score_func: str,
    reduction_func: str,
    n_features: int,
    random_state: int,
    cross_validate: bool,
    n_cpus: int,
):
    for ind_i, phenotype_file in enumerate(phenotypes_folder.glob("*.tsv")):
        phenotype_name = phenotype_file.stem.removesuffix("_phenotypes")
        if phenotype_name not in DATASETS:
            print(f"Skipping phenotype {phenotype_name}")
            continue
        for ind_j, feature in enumerate(FEATURES):
            feature_name = feature
            curr_feature_folder = features_folder / feature_name
            if not curr_feature_folder.is_dir():
                continue
            feature_file = (
                curr_feature_folder / f"{feature_name}_{phenotype_name}_features.tsv"
            )
            outputs_sub_dir = outputs_folder / feature_name
            outputs_sub_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"i={ind_i+1};j={ind_j+1}/{len(FEATURES)}. Predicting traits for {phenotype_name} using {feature_name}"
            )
            cmd = [
                "python",
                "-W",
                "ignore",
                "predict_traits.py",
                str(phenotype_file),
                str(feature_file),
                str(outputs_sub_dir),
                "--feature_type",
                feature_name,
                "--random_state",
                str(random_state),
                "--score_func",
                score_func,  # "f_classif", "chi2", "mutual_info_classif"
                "--reduction_func",
                reduction_func,  # "PCA", "NMF"
                "--n_features",
                str(n_features),
                "--n_cpus",
                str(n_cpus),
                # "--save_all",
            ]
            if cross_validate:
                cmd += "--cross_validate"
            print(f"Output folder: {outputs_sub_dir}")
            print(
                f"Score func: {score_func}, Reduction func: {reduction_func}, # of features: {n_features}"
            )
            print(f"Random state: {random_state}, Number of CPUs: {n_cpus}")
            subprocess.run(cmd)
            print("\n")


if __name__ == "__main__":
    # args = outputs_folder, score_func, reduction_func,  n_features, random_state, n_cpus
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outputs_folder", type=str, help="The folder to save the outputs"
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
    parser.add_argument("--random_state", type=int, default=42, help="Random state")
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
    # TODO: Add cross_validation parameter
    args = parser.parse_args()
    outputs_folder = pathlib.Path(args.outputs_folder)
    score_func = args.score_func
    reduction_func = args.reduction_func
    n_features = args.n_features
    random_state = args.random_state
    cross_validate = args.cross_validate
    n_cpus = args.n_cpus if args.n_cpus > 0 else mp.cpu_count()

    phenotypes_folder = pathlib.Path("data/processed/biolog/phenotypes/")
    features_folder = pathlib.Path("data/processed/biolog/features/")
    main(
        phenotypes_folder,
        features_folder,
        outputs_folder,
        score_func,
        reduction_func,
        n_features,
        random_state,
        cross_validate,
        n_cpus,
    )
