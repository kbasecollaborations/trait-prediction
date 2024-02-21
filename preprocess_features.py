#!/usr/bin/env python3

import gzip
import json
import pathlib

import tqdm

from trait_prediction.feature_selection.reduction import (
    remove_features_with_high_correlation,
    remove_features_with_low_variance,
)
from trait_prediction.utils import read_generic_features

DATASETS = ["ch", "pmi", "leaf"]
FEATURES = [
    "rast",
    "kofam",
    "kofam_modules",
    "uniref30",
    "uniref50",
    "uniref70",
    "uniref90",
    "uniprot_trembl",
    "cluster30",
    "cluster50",
    "cluster70",
    "cluster90",
    "eggnog_kegg",
    "eggnog_seed",
]


def main(
    features_folder: pathlib.Path,
    features_reduced_folder: pathlib.Path,
):
    pbar = tqdm.tqdm(FEATURES)
    for feature in pbar:
        for dataset in DATASETS:
            pbar.set_description(f"Processing {feature} for {dataset}")
            input_file = features_folder / f"{feature}/{feature}_{dataset}_features.tsv"
            output_folder = features_reduced_folder / feature
            output_folder.mkdir(parents=True, exist_ok=True)
            output_file = output_folder / f"{feature}_{dataset}_features_reduced.tsv"
            if output_file.is_file():
                continue
            low_var_file = output_folder / f"{feature}_{dataset}_low_var_features.txt"
            corr_file = output_folder / f"{feature}_{dataset}_corr_features.json.gz"
            if feature == "kofam_modules":
                bool_conversion = False
                dtype = "float64"
                var_thres = 0.0
            else:
                bool_conversion = True
                dtype = "uint8"
                var_thres = 0.01
            features = read_generic_features(
                input_file, bool_conversion=bool_conversion, dtype=dtype
            )
            features, low_var_features = remove_features_with_low_variance(
                features, threshold=var_thres
            )
            if features.shape[1] <= 50_000:
                corr_method = "numpy"
            else:
                corr_method = "numba_parallel"
            features, correlated_features_dict = remove_features_with_high_correlation(
                features, threshold=0.95, method=corr_method
            )
            features.to_csv(output_file, sep="\t", index=True)
            with open(low_var_file, "w") as fid:
                fid.write("\n".join(low_var_features))
            with gzip.open(corr_file, "wt") as gzfile:
                json.dump(correlated_features_dict, gzfile)


if __name__ == "__main__":
    data_folder = pathlib.Path("./data/processed/biolog/")
    features_folder = data_folder / "features/"
    features_reduced_folder = data_folder / "features_reduced/"
    main(features_folder, features_reduced_folder)
