#!/usr/bin/env python3

import argparse
import pathlib

import pandas as pd
from tqdm import tqdm

from trait_prediction.utils import read_generic_features


def create_dataset_phenotype_feature_map(
    datasets: list[str],
    feature_list: list[str],
    phenotype_list: list[pathlib.Path],
) -> dict[str, dict[str, dict[str, list[str]]]]:
    # dataset -> phenotype -> feature -> top 100 features
    map = dict()
    for dataset in datasets:
        map[dataset] = dict()
        for phenotype in phenotype_list:
            map[dataset][phenotype.stem] = dict()
            for feature in feature_list:
                map[dataset][phenotype.stem][feature] = []
    return map


def get_features(folder: pathlib.Path, n: int) -> list[str]:
    cv_feature_files = list(folder.glob("cv_shap_features_*.csv"))
    if not cv_feature_files:
        return []
    cv_feature_df = pd.concat(
        [pd.read_csv(f, index_col=0) for f in cv_feature_files], axis=1
    )
    features: list[str] = list(
        cv_feature_df.mean(axis=1).sort_values(ascending=False).index[:n]  # type: ignore
    )
    return features


def make_combined_feature_file(
    feature_map: dict[str, list[str]],
    phenotype_name: str,
    dataset_name: str,
    feature_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    dataset_name_nosuffix = dataset_name.removesuffix("_biolog")
    feature_df_list = []
    for feature_name, features in feature_map.items():
        feature_file = (
            feature_dir
            / f"{feature_name}/{feature_name}_{dataset_name_nosuffix}_features_reduced.tsv"
        )
        if not feature_file.is_file():
            raise ValueError(f"File {feature_file} does not exist")
        # NOTE: We do not include kofam_modules here
        bool_conversion = True
        dtype = "uint8"
        feature_df_list.append(
            read_generic_features(
                feature_file, bool_conversion=bool_conversion, dtype=dtype
            ).loc[:, features]
        )
    combined_features = pd.concat(feature_df_list, axis=1, join="inner")
    combined_features.to_csv(
        output_dir / f"{dataset_name_nosuffix}/{phenotype_name}_features_combined.tsv",
        sep="\t",
        index=True,
    )


def main(
    prediction_runs: list[pathlib.Path],
    feature_list: list[str],
    datasets: list[str],
    run_ids: list[str],
    feature_dir: pathlib.Path,
    output_dir: pathlib.Path,
) -> None:
    # Assumption: Only one run_id will ever be used
    for prediction_run in prediction_runs:
        if not prediction_run.is_dir():
            continue
        if prediction_run.stem not in run_ids:
            continue
        # dataset -> phenotype -> feature -> top 100 features
        phenotype_list = [
            d
            for d in (prediction_run / f"{feature_list[0]}/{datasets[0]}").iterdir()
            if d.is_dir()
        ]
        dataset_phenotype_feature_map = create_dataset_phenotype_feature_map(
            datasets, feature_list, phenotype_list
        )
        for feature in feature_list:
            feature_folder = prediction_run / feature
            if not feature_folder.is_dir():
                continue
            feature_name = feature
            for dataset_folder in feature_folder.iterdir():
                if not dataset_folder.is_dir():
                    continue
                if dataset_folder.stem not in datasets:
                    continue
                dataset_name = dataset_folder.stem
                for phenotype_folder in dataset_folder.iterdir():
                    phenotype_name = phenotype_folder.stem
                    if not phenotype_folder.is_dir():
                        continue
                    # NOTE: Here we take the top 100 features
                    selected_features = get_features(phenotype_folder, n=100)
                    if selected_features == []:
                        print(
                            f"No features found for {feature_name} in {phenotype_name}"
                        )
                    dataset_phenotype_feature_map[dataset_name][phenotype_name][
                        feature_name
                    ].extend(selected_features)
        for (
            dataset_name,
            phenotype_feature_map,
        ) in dataset_phenotype_feature_map.items():
            pbar = tqdm(phenotype_feature_map.items())
            for phenotype_name, feature_map in pbar:
                pbar.set_description(f"Processing {phenotype_name} for {dataset_name}")
                make_combined_feature_file(
                    feature_map, phenotype_name, dataset_name, feature_dir, output_dir
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine feature files")
    parser.add_argument(
        "input_dir",
        type=pathlib.Path,
        help="Directory containing the inputs",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Name of the dataset",
    )
    parser.add_argument(
        "run_id",
        type=str,
        help="Run ID",
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="data/processed/biolog/features_reduced/",
        help="Path to the feature directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/biolog/features_combined/",
        help="Run ID",
    )
    args = parser.parse_args()
    input_dir = pathlib.Path(args.input_dir)
    datasets = [args.dataset]
    run_ids = [args.run_id]
    output_dir = pathlib.Path(args.output_dir)
    feature_dir = pathlib.Path(args.feature_dir)
    prediction_runs = [fol for fol in input_dir.iterdir() if fol.is_dir()]
    feature_list = [
        "cluster30",
        "cluster50",
        "cluster70",
        "cluster90",
        "eggnog_kegg",
        # "eggnog_seed",
        "kofam",
        # "kofam_modules", # NOTE: Skipping this because it is float and also has only 140 features
        "rast",  # NOTE: This is reducing number of data points by 20
        # "uniprot_trembl", # NOTE: Skipping this because lots of genomes are missing
        "uniref30",
        "uniref50",
        "uniref70",
        # "uniref90", # NOTE: This has bad performance and fewer samples
    ]
    main(prediction_runs, feature_list, datasets, run_ids, feature_dir, output_dir)
