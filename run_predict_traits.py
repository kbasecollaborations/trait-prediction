#!/usr/bin/env python

import pathlib
import subprocess

DATASETS = ["ch", "pmi", "leaf"]
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
):
    for phenotype_file in phenotypes_folder.glob("*.tsv"):
        phenotype_name = phenotype_file.stem.removesuffix("_phenotypes")
        if phenotype_name != "ch":
            print(f"Skipping phenotype {phenotype_name}")
            continue
        for feature in FEATURES:
            feature_name = feature
            curr_feature_folder = features_folder / feature_name
            if not curr_feature_folder.is_dir():
                continue
            feature_file = (
                curr_feature_folder / f"{feature_name}_{phenotype_name}_features.tsv"
            )
            outputs_sub_dir = outputs_folder / feature_name
            outputs_sub_dir.mkdir(parents=True, exist_ok=True)
            print(f"Predicting traits for {phenotype_name} using {feature_name}")
            cmd = [
                "python",
                "-W",
                "ignore",
                "predict_traits.py",
                str(phenotype_file),
                str(feature_file),
                str(outputs_sub_dir),
                "--feature_type",
                "generic",
                "--random_state",
                "42",
            ]
            print(cmd)
            subprocess.run(cmd)


if __name__ == "__main__":
    phenotypes_folder = pathlib.Path("data/processed/biolog/phenotypes/")
    features_folder = pathlib.Path("data/processed/biolog/features/")
    outputs_folder = pathlib.Path("data/outputs/biolog/")
    main(phenotypes_folder, features_folder, outputs_folder)
