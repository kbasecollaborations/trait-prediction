#!/usr/bin/env python

import gc
import pathlib
from typing import Optional

import pandas as pd


def read_features(feature_file: pathlib.Path, feature_name: str) -> pd.DataFrame:
    with open(feature_file, "r") as fid:
        header = fid.readline().strip().split("\t")
    dtypes = dict()
    dtypes[header[0]] = "str"
    for col in header[1:]:
        if feature_name == "kofam_modules":
            dtypes[col] = "float64"
        else:
            dtypes[col] = "uint32"
    feature_chunks = pd.read_csv(
        feature_file, sep="\t", dtype=dtypes, iterator=True, chunksize=100
    )
    feature_df = pd.concat(feature_chunks, ignore_index=True)
    feature_df.set_index(header[0], inplace=True)
    feature_df.fillna(0, inplace=True)
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
    phenotype_file: pathlib.Path,
    phenotype_name: str,
    output_dir: pathlib.Path,
) -> Optional[pd.DataFrame]:
    file_prefix = f"{feature_name}_{phenotype_name}"
    output_sub_dir = output_dir / feature_name
    output_sub_dir.mkdir(parents=True, exist_ok=True)
    final_output_file = output_sub_dir / f"{file_prefix}_features.tsv"
    if final_output_file.exists() and final_output_file.is_file():
        print(f"Skipping {final_output_file} because it exists")
        return None
    phenotype_df = read_phenotypes(phenotype_file)
    rows = phenotype_df.index
    missing_rows = set(rows) - set(feature_df.index)
    if len(missing_rows) > 0:
        print(f"Missing {len(missing_rows)} keys for phenotype={phenotype_name}")
        missing_keys_file = output_sub_dir / f"{file_prefix}_missing_keys.txt"
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
    final_feature_df.to_csv(final_output_file, sep="\t", index=True)
    return final_feature_df


def main(
    phenotype_files: dict[str, pathlib.Path],
    feature_files: dict[str, pathlib.Path],
    output_dir: pathlib.Path,
):
    feature_data = {k: v for k, v in feature_files.items()}
    phenotype_data = {k: v for k, v in phenotype_files.items()}
    for feature_name, feature_file in feature_data.items():
        print("\n")
        print(f"Loading {feature_name} file")
        output_sub_dir = output_dir / feature_name
        if output_sub_dir.exists() and output_sub_dir.is_dir():
            print(f"Skipping {feature_name} because the data already exists")
            continue
        feature_df = read_features(feature_file, feature_name)
        for phenotype_name, phenotype_file in phenotype_data.items():
            print(
                f"Creating {feature_name} feature file for {phenotype_name} phenotype"
            )
            split_feature_data(
                feature_df, feature_name, phenotype_file, phenotype_name, output_dir
            )
        del feature_df
        gc.collect()


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
        "kofam": raw_feature_folder / "kofam-annotations.tsv",
        "kofam_modules": raw_feature_folder / "kofam-modules.tsv",
        "uniref30": raw_feature_folder / "uniref30-annotations.tsv",
        "uniref90": raw_feature_folder / "uniref90-annotations.tsv",
        "uniprot_trembl": raw_feature_folder / "uniprot_trembl-annotations.tsv",
        "cluster30": raw_feature_folder / "cluster-level-30.0.counts.tsv",
        "cluster50": raw_feature_folder / "cluster-level-50.0.counts.tsv",
        "cluster70": raw_feature_folder / "cluster-level-70.0.counts.tsv",
        "cluster90": raw_feature_folder / "cluster-level-90.0.counts.tsv",
        "eggnog_kegg": raw_feature_folder / "eggnog-annotations-kegg-ids.tsv",
        "eggnog_seed": raw_feature_folder / "eggnog-annotations-seed-orthologs.tsv",
    }
    output_dir = pathlib.Path("data/processed/biolog/features")
    main(phenotype_files, feature_files, output_dir)
