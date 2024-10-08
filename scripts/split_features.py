#!/usr/bin/env python

from pathlib import Path

import pandas as pd

from trait_prediction.main import Feature, FeatureIndex, FeatureInput


def index_format_func(x):
    return (
        x.strip()
        .split("?")[-1]
        .removesuffix(".RAST")
        .removesuffix(".fna")
        .removeprefix("g")
    )


def get_genomeids_from_phenotypes(phenotypes_folder: Path) -> dict[str, list[str]]:
    genomeid_map = {}
    for phenotype_file in phenotypes_folder.glob("*.tsv"):
        dataset = phenotype_file.stem.removesuffix("_phenotypes")
        phenotype_data = pd.read_csv(phenotype_file, sep="\t", index_col=0)
        genomeid_map[dataset] = list(phenotype_data.index.map(index_format_func))
    return genomeid_map


def split_features(
    input_folder: Path,
    output_folder: Path,
    genomeid_map: dict[str, list[str]],
    feature_name_subset: list[str],
) -> None:
    for feature_file in input_folder.glob("*.tsv"):
        feature_name = feature_file.stem
        if feature_name not in feature_name_subset:
            continue
        feature_index = FeatureIndex(name=feature_name, ftype="binary", dtype="uint8")
        feature_input = FeatureInput(
            path=feature_file, findex=feature_index, index_format_func=index_format_func
        )
        feature = Feature.read_data(feature_input)
        feature_data = feature.feature_data
        for dataset, genomeids in genomeid_map.items():
            feature_data_subset = feature_data.loc[genomeids]
            output_file = output_folder / f"{dataset}/{feature_name}.tsv"
            interim_output_folder = output_file.parent
            if not interim_output_folder.exists():
                interim_output_folder.mkdir(parents=True)
            feature_data_subset.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    input_folder = Path("../data/raw/features/biolog/")
    phenotypes_folder = Path("../data/raw/phenotypes")
    output_folder = Path("../data/interim/features")
    feature_name_subset = ["kofam"]
    genomeid_map = get_genomeids_from_phenotypes(phenotypes_folder)
    split_features(input_folder, output_folder, genomeid_map, feature_name_subset)
