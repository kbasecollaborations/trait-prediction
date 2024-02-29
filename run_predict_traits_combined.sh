#!/usr/bin/env bash

set -e

# Leaf
python -W ignore predict_traits.py \
	data/processed/biolog/phenotypes/leaf_phenotypes.tsv \
	data/processed/biolog/features_combined/leaf/ \
	data/outputs_combined_leaf/biolog_chi2_1000/combined/ \
	--feature_type "combined" \
	--score_func "None" \
	--reduction_func "None" \
	--n_features 1000 \
	--random_state 42 \
	--n_cpus 16 \
	--cross_validate

# CH
python -W ignore predict_traits.py \
	data/processed/biolog/phenotypes/ch_phenotypes.tsv \
	data/processed/biolog/features_combined/ch/ \
	data/outputs_combined_ch/biolog_chi2_1000/combined/ \
	--feature_type "combined" \
	--score_func "None" \
	--reduction_func "None" \
	--n_features 1000 \
	--random_state 42 \
	--n_cpus 16 \
	--cross_validate
