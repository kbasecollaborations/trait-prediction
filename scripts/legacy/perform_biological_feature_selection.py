#!/usr/bin/env python

import json
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from trait_prediction.main import Feature, FeatureIndex, FeatureInput, FeatureSet


def get_pathways_for_keyword(keyword: str) -> list[str]:
    search_url = f"http://rest.kegg.jp/find/pathway/{keyword}"
    response = requests.get(search_url)
    pathways = response.text.strip().split("\n")
    return [line.split("\t")[0] for line in pathways]


def get_modules_for_keyword(keyword: str) -> list[str]:
    search_url = f"http://rest.kegg.jp/find/module/{keyword}"
    response = requests.get(search_url)
    modules = response.text.strip().split("\n")
    return [line.split("\t")[0] for line in modules]


def get_orthologs_for_keyword(keyword: str) -> list[str]:
    search_url = f"http://rest.kegg.jp/find/orthology/{keyword}"
    response = requests.get(search_url)
    orthologs = response.text.strip().split("\n")
    return [line.split("\t")[0] for line in orthologs]


def get_modules_for_pathway(pathway_id: str) -> list[str]:
    # Retrieve pathway details to get modules
    get_url = f"http://rest.kegg.jp/get/{pathway_id}"
    response = requests.get(get_url)
    lines = [line.strip() for line in response.text.strip().split("\n")]
    modules = []
    for line in lines:
        if line.startswith("MODULE"):
            modules.append(line.split()[1])
        elif line.startswith("M"):
            modules.append(line.split()[0])
    return modules


def get_orthologs(keyword: str, module_mapping: dict[str, list[str]]) -> list[str]:
    orthologs = set()
    # Step1: Get orthologs from pathway
    pathways = get_pathways_for_keyword(keyword)
    for pathway_id in pathways:
        modules = get_modules_for_pathway(pathway_id)
        for raw_module_id in modules:
            module_id = raw_module_id.split(":")[-1]
            orthologs.update(module_mapping.get(module_id, []))
    # Step2: Get orthologs from modules
    for raw_module_id in get_modules_for_keyword(keyword):
        module_id = raw_module_id.split(":")[-1]
        orthologs.update(module_mapping.get(module_id, []))
    # Step3: Get orthologs from keyword
    for raw_ortholog_id in get_orthologs_for_keyword(keyword):
        ortholog_id = raw_ortholog_id.split(":")[-1]
        orthologs.add(ortholog_id)
    return sorted(orthologs)


def get_module_map(module_mapping_file: Path) -> dict[str, list[str]]:
    module_mapping_df = pd.read_csv(module_mapping_file, sep="\t", index_col=0)
    ko_list = (
        module_mapping_df["Definition"]
        .str.replace(" ", ",")
        .str.replace("+", ",")
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace("-", "")
        .str.split(",")
    )
    module_mapping = dict(zip(module_mapping_df.index, ko_list))
    return module_mapping


def create_phenotype_ko_map(
    keyword_map: dict[str, str], module_mapping: dict[str, list[str]], output_file: Path
) -> dict[str, list[str]]:
    output_folder = output_file.parent
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
    orthologs_map: dict[str, list[str]] = {}
    for phenotype_name, keyword in tqdm(keyword_map.items()):
        orthologs_map[phenotype_name] = get_orthologs(keyword, module_mapping)
    with open(output_file, "w") as file:
        json.dump(orthologs_map, file, indent=4)
    return orthologs_map


def index_format_func(x):
    return (
        x.strip()
        .split("?")[-1]
        .removesuffix(".RAST")
        .removesuffix(".fna")
        .removeprefix("g")
    )


def read_feature_data(
    input_data_folder: Path, datasets: list[str], feature_representation: str
) -> FeatureSet:
    features: list[Feature] = []
    for dataset in datasets:
        feature_file = input_data_folder / f"{dataset}/{feature_representation}.tsv"
        findex = FeatureIndex(name=dataset, ftype="binary", dtype="uint8")
        finput = FeatureInput(
            path=feature_file, findex=findex, index_format_func=index_format_func
        )
        features.append(Feature.read_data(finput))
    return FeatureSet(features)


def select_features(
    feature_set: FeatureSet, phenotype_ko_map: dict[str, list[str]]
) -> tuple[FeatureSet, dict[str, dict[str, list[str]]]]:
    features: list[Feature] = []
    missing_kos_map: dict[str, dict[str, list[str]]] = {}
    for feature in feature_set:
        feature_data_all = feature.feature_data
        feature_name = feature.findex.name
        missing_kos_map[feature_name] = {}
        for phenotype, ko_list in phenotype_ko_map.items():
            cols = feature_data_all.columns.intersection(ko_list)
            missing_kos_map[feature_name][phenotype] = sorted(set(ko_list) - set(cols))
            feature_data_sel = feature_data_all.loc[:, cols]
            name = f"{feature.findex.name}+{phenotype}"
            findex = FeatureIndex(name=name, ftype="binary", dtype="uint8")
            features.append(Feature(feature_data_sel, findex))
    return FeatureSet(features), missing_kos_map


def save_feature_files(feature_set: FeatureSet, output_folder: Path) -> None:
    for feature in feature_set:
        dataset, phenotype = feature.findex.name.split("+")
        feature_file = output_folder / f"{dataset}/{phenotype}.tsv"
        curr_output_folder = feature_file.parent
        if not curr_output_folder.exists():
            curr_output_folder.mkdir(parents=True)
        feature.feature_data.to_csv(feature_file, sep="\t", index=True)


if __name__ == "__main__":
    keyword_map = {
        "Alanine": "alanine",
        "Arginine": "arginine",
        "Cellobiose": "cellob",
        "Fructose": "fruct",
        "Galactose": "galac",
        "Galacturonic-Acid": "galactu",
        "Glucose": "glucose",
        "Glycerol": "glycerol",
        "Histidine": "histidine",
        "Maltose": "malt",
        "Mannitol": "mann",
        "Mannose": "mann",
        "Serine": "serine",
        "Sucrose": "sucr",
        "Trehalose": "trehal",
        "m-Inositol": "inositol",
    }
    datasets = ["atleaf", "lit", "pmi"]
    feature_representation = "kofam"
    module_mapping_file = Path("../data/external/mapping/module-definitions.tsv")
    output_folder = Path("../data/processed/features_selected")
    input_data_folder = Path("../data/interim/features")
    phenotype_ko_map_file = output_folder / "ko_features.json"
    module_mapping = get_module_map(module_mapping_file)
    if not phenotype_ko_map_file.exists():
        phenotype_ko_map = create_phenotype_ko_map(
            keyword_map, module_mapping, phenotype_ko_map_file
        )
    else:
        with open(phenotype_ko_map_file) as file:
            phenotype_ko_map = json.load(file)
    feature_set_full = read_feature_data(
        input_data_folder, datasets, feature_representation
    )
    feature_set_sel, missing_kos_map = select_features(
        feature_set_full, phenotype_ko_map
    )
    with open(output_folder / "missing_kos.json", "w") as file:
        json.dump(missing_kos_map, file, indent=4)
    save_feature_files(feature_set_sel, output_folder)
