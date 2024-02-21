#!/usr/bin/env bash

# This script is used to batch predict traits using the run_predict_traits.py script

################################################################################
## Commands for leaf
################################################################################

# Scoring: chi2=1000
python -W ignore run_predict_traits.py data/outputs_leaf/biolog_chi2_1000 \
	"leaf" \
	--score_func "chi2" \
	--n_features 1000 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
# Scoring: mutual_info_classif=1000
python -W ignore run_predict_traits.py data/outputs_leaf/biolog_mutual_info_1000 \
	"leaf" \
	--score_func "mutual_info_classif" \
	--n_features 1000 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
# Scoring: f_classif=1000
python -W ignore run_predict_traits.py data/outputs_leaf/biolog_f_1000 \
	"leaf" \
	--score_func "f_classif" \
	--n_features 1000 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48

################################################################################
# Reduction: NMF=200
python -W ignore run_predict_traits.py data/outputs_leaf/biolog_nmf_200 \
	"leaf" \
	--reduction_func "NMF" \
	--n_features 200 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
# Reduction: PCA=200
python -W ignore run_predict_traits.py data/outputs_leaf/biolog_pca_200 \
	"leaf" \
	--reduction_func "PCA" \
	--n_features 200 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48

################################################################################
# All features
python -W ignore run_predict_traits.py data/outputs_leaf/biolog \
	"leaf" \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48

################################################################################
## Commands for ch
################################################################################

# Scoring: chi2=1000
python -W ignore run_predict_traits.py data/outputs_ch/biolog_chi2_1000 \
	"ch" \
	--score_func "chi2" \
	--n_features 1000 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
# Scoring: mutual_info_classif=1000
python -W ignore run_predict_traits.py data/outputs_ch/biolog_mutual_info_1000 \
	"ch" \
	--score_func "mutual_info_classif" \
	--n_features 1000 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
# Scoring: f_classif=1000
python -W ignore run_predict_traits.py data/outputs_ch/biolog_f_1000 \
	"ch" \
	--score_func "f_classif" \
	--n_features 1000 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48

################################################################################
# Reduction: NMF=200
python -W ignore run_predict_traits.py data/outputs_ch/biolog_nmf_200 \
	"ch" \
	--reduction_func "NMF" \
	--n_features 200 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
# Reduction: PCA=200
python -W ignore run_predict_traits.py data/outputs_ch/biolog_pca_200 \
	"ch" \
	--reduction_func "PCA" \
	--n_features 200 \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48

################################################################################
# All features
python -W ignore run_predict_traits.py data/outputs_ch/biolog \
	"ch" \
	--cross_validate \
	--random_state 42 \
	--n_cpus 48
