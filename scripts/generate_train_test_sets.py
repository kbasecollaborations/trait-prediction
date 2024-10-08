#!/usr/bin/env python

from pathlib import Path

import numpy as np
import pandas as pd

from trait_prediction.main import (
    Feature,
    FeatureIndex,
    FeatureInput,
    FeatureSet,
    Phenotype,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)


def index_format_func(x):
    return (
        x.strip()
        .split("?")[-1]
        .removesuffix(".RAST")
        .removesuffix(".fna")
        .removeprefix("g")
    )


def get_genome_info(data_folder: Path) -> pd.DataFrame:
    # Load the genome to class mapping
    genome_info_file = data_folder / "raw/phylogeny/genome_info_parsed.tsv"
    genome_info_df = pd.read_csv(genome_info_file, sep="\t", index_col=0)
    return genome_info_df


def load_phenotypes(
    data_folder: Path, datasets: list[str], genome_info_df: pd.DataFrame
) -> PhenotypeSet:
    phenotypes_list: list[Phenotype] = []
    # Find common phenotypes across the datasets
    phenotype_name_sets: list[set[str]] = []
    for dataset in datasets:
        phenotype_folder = data_folder / f"processed/phenotypes/{dataset}"
        phenotype_name_sets.append({f.stem for f in phenotype_folder.glob("*.tsv")})
    phenotype_names_common = set.intersection(*phenotype_name_sets)
    # Read the phenotypes
    for phenotype_name in phenotype_names_common:
        for dataset in datasets:
            suffix = dataset.split("-")[-1] if "-" in dataset else ""
            dataset_names = dataset.split("-")[0].split("+")
            pinputs: list[PhenotypeInput] = []
            for dataset_name in dataset_names:
                phenotype_folder = data_folder / f"processed/phenotypes/{dataset_name}"
                phenotype_file = phenotype_folder / f"{phenotype_name}.tsv"
                pindex = PhenotypeIndex(name=phenotype_name, category=dataset_name)
                pinput = PhenotypeInput(
                    path=phenotype_file,
                    pindex=pindex,
                    index_format_func=index_format_func,
                )
                pinputs.append(pinput)
            if len(pinputs) > 1:
                phenotype = Phenotype.merge_data(tuple(pinputs))
            else:
                pinput = pinputs[0]
                phenotype = Phenotype.read_data(pinput)
            # Remove aprropriate elements from the phenotype data
            if suffix:
                if suffix == "g":
                    class_name = "Gammaproteobacteria"
                elif suffix == "a":
                    class_name = "Alphaproteobacteria"
                else:
                    raise ValueError(f"Unknown suffix: {suffix}")
                phenotype_data_all = phenotype.phenotype_data
                pindex = phenotype.pindex
                drop_indices = list(
                    genome_info_df[genome_info_df["Class"] == class_name].index
                )
                phenotype_data = phenotype_data_all.drop(drop_indices, errors="ignore")
                phenotype = Phenotype(phenotype_data, pindex)
            phenotypes_list.append(phenotype)
    return PhenotypeSet(phenotypes_list)


def load_train_features(
    data_folder: Path,
    datasets: list[str],
    feature_representation: str,
    feature_type: str,
    genome_info_df: pd.DataFrame,
) -> FeatureSet:
    features_list: list[Feature] = []
    for dataset in datasets:
        suffix = dataset.split("-")[-1] if "-" in dataset else ""
        dataset_names = dataset.split("-")[0].split("+")
        finputs: list[FeatureInput] = []
        for dataset_name in dataset_names:
            feature_folder = data_folder / f"interim/features/{dataset_name}"
            feature_file = feature_folder / f"{feature_representation}.tsv"
            feature_name = f"{feature_representation}.{feature_type}"
            findex = FeatureIndex(name=feature_name, ftype="binary", dtype="uint8")
            finput = FeatureInput(
                path=feature_file,
                findex=findex,
                index_format_func=index_format_func,
            )
            finputs.append(finput)
        if len(finputs) > 1:
            feature = Feature.merge_data(tuple(finputs))
        else:
            finput = finputs[0]
            feature = Feature.read_data(finput)
        # Remove aprropriate elements from the feature data
        if suffix:
            if suffix == "g":
                class_name = "Gammaproteobacteria"
            elif suffix == "a":
                class_name = "Alphaproteobacteria"
            else:
                raise ValueError(f"Unknown suffix: {suffix}")
            feature_data_all = feature.feature_data
            findex = feature.findex
            drop_indices = list(
                genome_info_df[genome_info_df["Class"] == class_name].index
            )
            feature_data = feature_data_all.drop(drop_indices, errors="ignore")
            feature = Feature(feature_data, findex)
        features_list.append(feature)
    return FeatureSet(features_list)


def read_feature_cols(
    data_folder: Path,
    datasets: list[str],
    feature_representation: str,
    feature_types: list[str],
) -> dict[str, dict[str, list[str]]]:
    pass


if __name__ == "__main__":
    # Parameters
    random_seed = 42
    np.random.seed(random_seed)
    datasets = ["atleaf", "lit", "pmi"]
    datasets_train_all = [
        "atleaf",
        "lit",
        "pmi",
        "atleaf+lit",
        "atleaf+lit-g",
        "atleaf+lit-a",
        "atleaf+lit+pmi",
        "atleaf+lit+pmi-g",
        "atleaf+lit+pmi-a",
    ]
    datasets_test_all = [
        "atleaf",
        "lit",
        "pmi",
        "out_gamma",
        "out_alpha",
        "in_abb",
        "uniform",
    ]
    train_test_map = {
        "atleaf": ["in_abb", "lit", "out_gamma", "pmi", "uniform"],
        "lit": ["atleaf", "in_abb", "out_alpha", "pmi", "uniform"],
        "atleaf_lit": ["in_abb", "pmi"],
        "atleaf_lit-g": ["out_gamma", "pmi"],
        "atleaf_lit-a": ["out_alpha", "pmi"],
        "atleaf_lit_pmi": [],
        "atleaf_lit_pmi-g": ["out_gamma"],
        "atleaf_lit_pmi-a": ["out_alpha"],
    }
    # NOTE: Final folder feat(kofam_full)_train(atleaf+lit)_test(in_abb)
    feature_representation = "kofam"
    feature_types = ["full", "nocorr", "sel"]
    data_folder = Path("../data")
    # Data loading functions
    genome_info_df = get_genome_info(data_folder)
    full_phenotype_set: PhenotypeSet = load_phenotypes(
        data_folder, datasets_train_all, genome_info_df
    )
    train_feature_set: FeatureSet = load_train_features(
        data_folder, datasets_train_all, feature_representation, "full", genome_info_df
    )
    feature_type_col_dict = read_feature_cols(
        data_folder, datasets, feature_representation, feature_types
    )
    # Create test sets
    test_phenotype_set, test_feature_set = create_test_data()
    # Train test splits
    # Data processing functions
