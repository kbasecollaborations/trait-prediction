#!/usr/bin/env python

import pathlib

import pandas as pd

from trait_prediction.utils import read_generic_features


def read_features(feature_file: pathlib.Path) -> pd.DataFrame:
    feature_df = read_generic_features(feature_file)
    feature_df.index = [r.rstrip(".RAST") for r in feature_df.index]  # type: ignore
    feature_df.index.name = "genomeID"  # type: ignore
    return feature_df


def read_phenotypes(phenotype_file: pathlib.Path) -> pd.DataFrame:
    phenotype_df = pd.read_csv(
        phenotype_file, sep="\t", index_col=0, dtype={"genomeID": str}
    ).astype("Int64")
    return phenotype_df


def split_feature_data(
    feature_df: pd.DataFrame,
    feature_name: str,
    phenotype_df: pd.DataFrame,
    phenotype_name: str,
    output_dir: pathlib.Path,
) -> pd.DataFrame:
    rows = phenotype_df.index
    file_prefix = f"{feature_name}_{phenotype_name}"
    missing_rows = set(rows) - set(feature_df.index)
    print(missing_rows)
    if len(missing_rows) > 0:
        print(f"Missing {len(missing_rows)} keys for phenotype={phenotype_name}")
        missing_keys_file = output_dir / f"{file_prefix}_missing_keys.txt"
        with open(missing_keys_file, "w") as fid:
            for missing_row in missing_rows:
                fid.write(missing_row)
                fid.write("\n")
        selected_rows = list(set(rows).intersection(set(feature_df.index)))
    else:
        selected_rows = rows
    new_feature_df = feature_df.loc[selected_rows, :]
    new_feature_df.index.name = "genomeID"
    final_feature_df = new_feature_df.loc[
        new_feature_df.any(axis=1), new_feature_df.any(axis=0)
    ]
    final_output_file = output_dir / f"{file_prefix}_features.tsv"
    final_feature_df.to_csv(final_output_file, sep="\t", index=True)
    return final_feature_df


def main(
    phenotype_files: dict[str, pathlib.Path],
    feature_files: dict[str, pathlib.Path],
    output_dir: pathlib.Path,
):
    feature_data = {k: read_features(v) for k, v in feature_files.items()}
    phenotype_data = {k: read_phenotypes(v) for k, v in phenotype_files.items()}
    for feature_name, feature_df in feature_data.items():
        for phenotype_name, phenotype_df in phenotype_data.items():
            print(
                f"Creating {feature_name} feature file for {phenotype_name} phenotype"
            )
            split_feature_data(
                feature_df, feature_name, phenotype_df, phenotype_name, output_dir
            )


if __name__ == "__main__":
    phenotypes_folder = pathlib.Path("data/processed/biolog/phenotypes/")
    raw_feature_folder = pathlib.Path("data/raw/biolog/features/")
    phenotype_files = {
        "ch": phenotypes_folder / "ch_phenotypes.tsv",
        "leaf": phenotypes_folder / "leaf_phenotypes.tsv",
        "pmi": phenotypes_folder / "pmi_phenotypes.tsv",
    }
    feature_files = {
        "rast": raw_feature_folder / "rast-annotations.tsv",
    }
    output_dir = pathlib.Path("data/processed/biolog/features")
    main(phenotype_files, feature_files, output_dir)
