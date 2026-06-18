#!/usr/bin/env python

import json
from pathlib import Path

data_folder = Path("../data/processed/train_test_sets_v2")

# Expected # of results
n_feature_reps = 1
n_feature_types = 3
n_phenotypes = 16
train_test_map = {
    "atleaf": ["in_abb", "lit", "out_gamma", "pmi", "uniform"],
    "lit": ["atleaf", "in_abb", "out_alpha", "pmi", "uniform"],
    "atleaf+lit": ["in_abb", "pmi", "uniform"],
    "atleaf+lit-g": ["out_gamma", "pmi", "uniform"],
    "atleaf+lit-a": ["out_alpha", "pmi", "uniform"],
    "atleaf+lit+pmi": ["uniform"],
    "atleaf+lit+pmi-g": ["out_gamma", "uniform"],
    "atleaf+lit+pmi-a": ["out_alpha", "uniform"],
}
train_test_combinations = [1 for test_ids in train_test_map.values() for _ in test_ids]
n_train_test_combinations = sum(train_test_combinations)
n_reps = 5
n_expected_results = (
    n_feature_reps * n_feature_types * n_phenotypes * n_train_test_combinations * n_reps
)

# 1. Check if there are any missing results.json files
n_combinations = len([d for d in data_folder.iterdir() if d.is_dir()])
n_results = len([d for d in data_folder.glob("**/results.json") if d.is_file()])
assert 5 * n_combinations == n_results == n_expected_results

# 2. Verify that # of results.json files == len of data_map.json
with open(data_folder / "data_map.json") as fid:
    data_map = json.load(fid)
# assert len(data_map) == n_results
keys = [
    "feature_name",
    "feature_type",
    "phenotype_name",
    "train_set_id",
    "test_set_id",
    "rep",
]
uniques = list(set(frozenset([l[k] for k in keys]) for l in data_map))
print(len(uniques), n_results)
print(len(data_map) - len(uniques))
# FIXME: There are issues with this the data_map.json file

# 3. For every train and test set make sure that the train_indices and test_indices don't overlap
for comb_dir in data_folder.iterdir():
    if not comb_dir.is_dir():
        continue
    for rep_dir in comb_dir.iterdir():
        if not rep_dir.is_dir():
            continue
        train_ind_file = rep_dir / "train_indices.txt"
        with open(train_ind_file) as fid:
            train_indices = set(fid.read().splitlines())
        test_ind_file = rep_dir / "test_indices.txt"
        with open(test_ind_file) as fid:
            test_indices = set(fid.read().splitlines())
        assert train_indices.isdisjoint(test_indices)
        assert len(train_indices.intersection(test_indices)) == 0
