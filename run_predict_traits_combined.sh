#!/usr/bin/env bash

python -W ignore predict_traits.py \
	data/processed/biolog/phenotypes/leaf_phenotypes.tsv \
	data/processed/biolog/features_combined/leaf/ \
	data/outputs_combined/biolog_mutual_info_1000/combined/ \
	--feature_type "combined" \
	--score_func "None" \
	--reduction_func "None" \
	--n_features 1000 \
	--random_state 42 \
	--n_cpus 16 \
	--cross_validate
