#!/usr/bin/env python

import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef

from trait_prediction.main import (
    DataSet,
    Feature,
    FeatureIndex,
    FeatureInput,
    FeatureSet,
    Phenotype,
    PhenotypeIndex,
    PhenotypeInput,
    PhenotypeSet,
)


@dataclass(frozen=True)
class TrainsetIndex:
    feature_name: str  # eg: kofam
    feature_type: str  # eg: full
    phenotype_name: str  # eg: Alanine
    train_set_id: str  # eg: atleaf+lit


@dataclass(frozen=True)
class TrainTestIndex:
    feature_name: str  # eg: kofam
    feature_type: str  # eg: full
    phenotype_name: str  # eg: Alanine
    train_set_id: str  # eg: atleaf+lit
    test_set_id: str  # eg: in_abb
    rep: int  # 0-n_rep


@dataclass
class TrainTestData:
    index: TrainTestIndex
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    output_folder: Path


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
    for dataset in ["atleaf", "lit", "pmi"]:
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
                phenotype_data = phenotype.phenotype_data
                pindex = PhenotypeIndex(name=phenotype.pindex.name, category=dataset)
                phenotype = Phenotype(phenotype_data, pindex)
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
                pindex = PhenotypeIndex(name=phenotype.pindex.name, category=dataset)
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
    genome_info_df: pd.DataFrame,
) -> FeatureSet:
    features_list: list[Feature] = []
    for dataset in datasets:
        suffix = dataset.split("-")[-1] if "-" in dataset else ""
        dataset_names = dataset.split("-")[0].split("+")
        finputs: list[FeatureInput] = []
        for dataset_name in dataset_names:
            feature_folder = data_folder / f"interim/features_mod/{dataset_name}"
            feature_file = feature_folder / f"{feature_representation}.tsv"
            findex = FeatureIndex(name=dataset_name, ftype="binary", dtype="uint8")
            finput = FeatureInput(
                path=feature_file,
                findex=findex,
                index_format_func=index_format_func,
            )
            finputs.append(finput)
        if len(finputs) > 1:
            feature = Feature.merge_data(tuple(finputs))
            feature_data = feature.feature_data
            findex = FeatureIndex(name=dataset, ftype="binary", dtype="uint8")
            feature = Feature(feature_data, findex)
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
            findex = FeatureIndex(name=dataset, ftype="binary", dtype="uint8")
            drop_indices = list(
                genome_info_df[genome_info_df["Class"] == class_name].index
            )
            feature_data = feature_data_all.drop(drop_indices, errors="ignore")
            feature = Feature(feature_data, findex)
        features_list.append(feature)
    return FeatureSet(features_list)


def _get_cols(feature_file: Path) -> set[str]:
    with open(feature_file, "r") as fid:
        cols = fid.readline().strip().split("\t")[1:]
    assert "genomeID" not in cols
    return set(cols)


def read_feature_cols(
    data_folder: Path,
    datasets: list[str],
    feature_representation: str,
    feature_types: list[str],
    phenotype_names: list[str],
) -> dict[TrainsetIndex, list[str]]:
    selected_feat_map: dict[TrainsetIndex, list[str]] = dict()
    for dataset in datasets:
        for feature_type in feature_types:
            for phenotype_name in phenotype_names:
                dataset_names = dataset.split("-")[0].split("+")
                feature_cols: list[set] = []
                for dataset_name in dataset_names:
                    if feature_type == "full":
                        feature_file = (
                            data_folder
                            / f"interim/features/{dataset_name}/{feature_representation}.tsv"
                        )
                    elif feature_type == "nocorr":
                        feature_file = (
                            data_folder
                            / f"processed/features_reduced/{dataset_name}/{feature_representation}/{feature_representation}.tsv"
                        )
                    elif feature_type == "sel":
                        feature_folder = (
                            data_folder
                            / f"processed/features_selected/{dataset_name}/{feature_representation}"
                        )
                        feature_file = feature_folder / f"{phenotype_name}.tsv"
                    else:
                        raise ValueError(f"Unknown feature type: {feature_type}")
                    feature_cols.append(_get_cols(feature_file))
                key = TrainsetIndex(
                    feature_representation, feature_type, phenotype_name, dataset
                )
                selected_feat_map[key] = sorted(set.union(*feature_cols))
    return selected_feat_map


def _make_classifier():
    clf = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=1)
    return clf


def _get_feature_importances(clf, n: int = 100) -> pd.Series:
    feature_importances = clf.feature_importances_
    feature_importances = (
        pd.Series(feature_importances, index=clf.feature_names_in_)
        .sort_values(ascending=False)
        .head(n)
    )
    feature_importances.name = "Importance"
    feature_importances.index.name = "Feature"
    return feature_importances


def train_and_score(
    train_test_data: TrainTestData,
) -> tuple[dict[str, float], pd.Series]:
    clf = _make_classifier()
    default_scores = {"acc": -1.0, "bacc": -1.0, "mcc": -1.0}
    X_train, y_train, X_test, y_test = (
        train_test_data.X_train,
        train_test_data.y_train,
        train_test_data.X_test,
        train_test_data.y_test,
    )
    if X_train.shape[0] < 50:
        return default_scores, pd.Series()
    # Class counts
    train_class0_count = np.sum(y_train == 0)
    train_class1_count = np.sum(y_train == 1)
    test_class0_count = np.sum(y_test == 0)
    test_class1_count = np.sum(y_test == 1)
    if min(train_class0_count, train_class1_count) < 10:
        return default_scores, pd.Series()
    # Train
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # Scores
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    scores = {
        "acc": acc,
        "bacc": bacc,
        "mcc": mcc,
        "train_class0_count": train_class0_count,
        "train_class1_count": train_class1_count,
        "test_class0_count": test_class0_count,
        "test_class1_count": test_class1_count,
    }
    feature_importances = _get_feature_importances(clf)
    return scores, feature_importances


def _get_class_indices(
    genome_info_df: pd.DataFrame, query_classes: list[str]
) -> pd.Index:
    class_index = genome_info_df[genome_info_df["Class"].isin(query_classes)].index
    return class_index


def _get_uniform_indices(
    feature_df: pd.DataFrame, distance_df: pd.DataFrame, n_reps: int, test_frac: float
) -> pd.Index:
    common_index = feature_df.index.intersection(distance_df.index)
    distance_df_subset = distance_df.loc[common_index, common_index]
    n_clusters = int(len(common_index) * test_frac)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    clustering.fit(distance_df_subset)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    sampled_elements: list[str] = []
    for _ in range(n_reps):
        sampled_elements.extend(
            [
                np.random.choice(common_index[labels == unique_label], 1)[0]
                for unique_label in unique_labels
            ]
        )
    return pd.Index(sampled_elements)


def create_train_test_sets(
    full_dataset: DataSet,
    train_test_map: dict[str, list[str]],
    genome_info_df: pd.DataFrame,
    feature_types: list[str],
    selected_feat_map: dict[TrainsetIndex, list[str]],
    feature_representation: str,
    distance_df: pd.DataFrame,
    n_reps: int,
    test_frac: float,
    output_folder: Path,
) -> Iterator[TrainTestData]:
    phenotype_names = [p.pindex.name for p in full_dataset.phenotypes]
    for trainset_name, testset_names in train_test_map.items():
        for testset_name in testset_names:
            print(f"Creating data set for train({trainset_name})+test({testset_name})")
            for phenotype_name in phenotype_names:
                pindex_train = PhenotypeIndex(
                    name=phenotype_name, category=trainset_name
                )
                findex_train = FeatureIndex(
                    name=trainset_name, ftype="binary", dtype="uint8"
                )
                phenotype_train, feature_train = full_dataset.get_data(
                    pindex_train, findex_train
                )
                pindex_3datasets = PhenotypeIndex(
                    name=phenotype_name, category="atleaf+lit+pmi"
                )
                findex_3datasets = FeatureIndex(
                    name="atleaf+lit+pmi", ftype="binary", dtype="uint8"
                )
                phenotype_3datasets, feature_3datasets = full_dataset.get_data(
                    pindex_3datasets, findex_3datasets
                )
                # TODO: Refactor this to a new function
                if testset_name in ["atleaf", "lit", "pmi"]:
                    pindex_test = PhenotypeIndex(
                        name=phenotype_name, category=testset_name
                    )
                    findex_test = FeatureIndex(
                        name=testset_name, ftype="binary", dtype="uint8"
                    )
                    _, feature_test = full_dataset.get_data(pindex_test, findex_test)
                    indices_to_sample = feature_test.feature_data.index
                elif testset_name == "out_gamma":
                    class_index = _get_class_indices(
                        genome_info_df, ["Gammaproteobacteria"]
                    )
                    indices_to_sample = (
                        feature_3datasets.feature_data.index.intersection(class_index)
                    )
                elif testset_name == "out_alpha":
                    class_index = _get_class_indices(
                        genome_info_df, ["Alphaproteobacteria"]
                    )
                    indices_to_sample = (
                        feature_3datasets.feature_data.index.intersection(class_index)
                    )
                elif testset_name == "in_abb":
                    class_index = _get_class_indices(
                        genome_info_df, ["Bacilli", "Bacteroidia", "Actinomycetia"]
                    )
                    indices_to_sample = (
                        feature_3datasets.feature_data.index.intersection(class_index)
                    )
                elif testset_name == "uniform":
                    indices_to_sample = _get_uniform_indices(
                        feature_3datasets.feature_data, distance_df, n_reps, test_frac
                    )
                else:
                    raise ValueError(f"Unknown testset_name: {testset_name}")
                n_samples = int(len(indices_to_sample) * test_frac)
                test_indices = [
                    np.random.choice(indices_to_sample, n_samples, replace=False)
                    for _ in range(n_reps)
                ]
                train_indices = [
                    feature_train.feature_data.index.difference(test_index)
                    for test_index in test_indices
                ]
                # Apply feature selection ["full", "nocorr", "sel"]
                feature_cols: dict[str, list[str]] = dict()
                for feature_type in feature_types:
                    key = TrainsetIndex(
                        feature_representation,
                        feature_type,
                        phenotype_name,
                        trainset_name,
                    )
                    feature_cols[feature_type] = selected_feat_map[key]
                for rep in range(n_reps):
                    for feature_type in feature_types:
                        key = TrainTestIndex(
                            feature_name=feature_representation,
                            feature_type=feature_type,
                            phenotype_name=phenotype_name,
                            train_set_id=trainset_name,
                            test_set_id=testset_name,
                            rep=rep,
                        )
                        train_rows = train_indices[rep]
                        test_rows = test_indices[rep]
                        cols = feature_cols[feature_type]
                        cols.append("dataset")
                        X_train_full = feature_train.feature_data
                        y_train_full = phenotype_train.phenotype_data
                        X_3datasets = feature_3datasets.feature_data
                        y_3datasets = phenotype_3datasets.phenotype_data
                        # Add checks to ensure that the data is valid
                        assert (X_train_full.index == y_train_full.index).all()
                        if feature_type != "full":
                            assert len(cols) < X_train_full.shape[1]
                        X_train = X_train_full.loc[train_rows, cols]
                        y_train = y_train_full.loc[train_rows]
                        X_test = X_3datasets.loc[test_rows, cols]
                        y_test = y_3datasets.loc[test_rows]
                        train_test_data = TrainTestData(
                            index=key,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            output_folder=output_folder,
                        )
                        yield train_test_data


def save_train_test_sets(
    train_test_data: TrainTestData, output_folder: Path, skip: bool
) -> Path | None:
    output_folder.mkdir(parents=True, exist_ok=True)
    key = train_test_data.index
    feature_name = key.feature_name
    feature_type = key.feature_type
    phenotype_name = key.phenotype_name
    train_set_id = key.train_set_id
    test_set_id = key.test_set_id
    rep = key.rep
    folder_prefix = f"feat({feature_name}_{feature_type})-pheno({phenotype_name})"
    folder_suffix = f"train({train_set_id})-test({test_set_id})"
    curr_folder = output_folder / f"{folder_prefix}-{folder_suffix}" / f"{rep}"
    if skip and curr_folder.is_dir():
        return None
    curr_folder.mkdir(parents=True, exist_ok=True)
    X_train, _, X_test, _ = (
        train_test_data.X_train,
        train_test_data.y_train,
        train_test_data.X_test,
        train_test_data.y_test,
    )
    # NOTE: Saving only the rows and cols
    train_indices = X_train.index
    with open(curr_folder / "train_indices.txt", "w") as fid:
        fid.write("\n".join(train_indices))
    test_indices = X_test.index
    with open(curr_folder / "test_indices.txt", "w") as fid:
        fid.write("\n".join(test_indices))
    feature_columns = X_train.columns
    with open(curr_folder / "feature_columns.txt", "w") as fid:
        fid.write("\n".join(feature_columns))
    return curr_folder


def run_task(train_test_data: TrainTestData) -> dict:
    key = train_test_data.index
    output_folder = train_test_data.output_folder
    curr_folder = save_train_test_sets(train_test_data, output_folder, skip=True)
    if curr_folder is None:
        return dict()
    scores, feature_importances = train_and_score(train_test_data)
    results = {
        "feature_name": key.feature_name,
        "feature_type": key.feature_type,
        "phenotype_name": key.phenotype_name,
        "train_set_id": key.train_set_id,
        "test_set_id": key.test_set_id,
        "rep": key.rep,
        "path": str(curr_folder),
        "accuracy": scores["acc"],
        "balanced_accuracy": scores["bacc"],
        "matthews_corrcoef": scores["mcc"],
    }
    # FIXME: There is a bug here, python is also writing to script directory for these two files
    with open(curr_folder / "results.json", "w") as fid:
        fid.write(json.dumps(results))
    feature_importances.to_csv(
        curr_folder / "feature_importances.tsv",
        sep="\t",
        index=True,
    )
    return results


if __name__ == "__main__":
    # Parameters
    random_seed = 42
    np.random.seed(random_seed)
    n_reps = 5
    test_frac = 0.2
    output_folder = Path("../data/processed/train_test_sets_v2")
    datasets = ["atleaf", "lit", "pmi"]
    datasets_train_all = [
        "atleaf",
        "lit",
        "pmi",
        "atleaf-a",
        "lit-g",
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
        "atleaf-a": ["in_abb", "lit", "out_alpha", "out_gamma", "pmi", "uniform"],
        "lit": ["atleaf", "in_abb", "out_alpha", "pmi", "uniform"],
        "lit-g": ["atleaf", "in_abb", "out_alpha", "out_gamma", "pmi", "uniform"],
        "atleaf+lit": ["in_abb", "pmi", "uniform"],
        "atleaf+lit-g": ["out_gamma", "pmi", "uniform"],
        "atleaf+lit-a": ["out_alpha", "pmi", "uniform"],
        "atleaf+lit+pmi": ["uniform"],
        "atleaf+lit+pmi-g": ["out_gamma", "uniform"],
        "atleaf+lit+pmi-a": ["out_alpha", "uniform"],
    }
    # Pairwise distance data
    distance_matrix_file = Path("../data/processed/distance_matrix.tsv")
    distance_df = pd.read_csv(distance_matrix_file, sep="\t", index_col=0)
    # NOTE: Final folder feat(kofam_full)-pheno(Alanine)-train(atleaf+lit)-test(in_abb)
    feature_representation = "kofam"
    feature_types = ["full", "nocorr", "sel"]
    data_folder = Path("../data")
    # Data loading functions
    genome_info_df = get_genome_info(data_folder)
    full_phenotypeset: PhenotypeSet = load_phenotypes(
        data_folder, datasets_train_all, genome_info_df
    )
    phenotype_names = [phenotype.pindex.name for phenotype in full_phenotypeset]
    full_featureset: FeatureSet = load_train_features(
        data_folder, datasets_train_all, feature_representation, genome_info_df
    )
    full_dataset = DataSet(full_phenotypeset, full_featureset)
    selected_feat_map = read_feature_cols(
        data_folder,
        datasets_train_all,
        feature_representation,
        feature_types,
        phenotype_names,
    )
    # Create train and test sets
    # NOTE: We are not using an eval set
    task_list = create_train_test_sets(
        full_dataset,
        train_test_map,
        genome_info_df,
        feature_types,
        selected_feat_map,
        feature_representation,
        distance_df,
        n_reps=n_reps,
        test_frac=test_frac,
        output_folder=output_folder,
    )
    data_map = []
    with mp.Pool(processes=20) as pool:
        results = pool.imap(run_task, task_list)
        for result in results:
            data_map.append(result)
    with open(output_folder / "data_map.json", "w") as f:
        json.dump(data_map, f)
