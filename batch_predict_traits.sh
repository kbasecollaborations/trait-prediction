#!/usr/bin/env bash

# This script is used to batch predict traits using the run_predict_traits.py script

## Commands

# Scoring: chi2=1000
python -W ignore run_predict_traits.py data/outputs/biolog_chi2_1000 \
    --score_func "chi2" \
    --n_features 1000 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
# Scoring: mutual_info_classif=1000
python -W ignore run_predict_traits.py data/outputs/biolog_mutual_info_1000 \
    --score_func "mutual_info_classif" \
    --n_features 1000 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
# Scoring: f_classif=1000
python -W ignore run_predict_traits.py data/outputs/biolog_f_1000 \
    --score_func "f_classif" \
    --n_features 1000 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16

# Reduction: NMF=200
python -W ignore run_predict_traits.py data/outputs/biolog_nmf_200 \
    --reduction_func "NMF" \
    --n_features 200 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
# Reduction: PCA=200
python -W ignore run_predict_traits.py data/outputs/biolog_pca_200 \
    --reduction_func "PCA" \
    --n_features 200 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16

# Scoring: chi2=500
python -W ignore run_predict_traits.py data/outputs/biolog_chi2_500 \
    --score_func "chi2" \
    --n_features 500 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
# Scoring: mutual_info_classif=500
python -W ignore run_predict_traits.py data/outputs/biolog_mutual_info_500 \
    --score_func "mutual_info_classif" \
    --n_features 500 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
# Scoring: f_classif=500
python -W ignore run_predict_traits.py data/outputs/biolog_f_500 \
    --score_func "f_classif" \
    --n_features 500 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16

# Reduction: NMF=100
python -W ignore run_predict_traits.py data/outputs/biolog_nmf_100 \
    --reduction_func "NMF" \
    --n_features 100 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
# Reduction: PCA=100
python -W ignore run_predict_traits.py data/outputs/biolog_pca_100 \
    --reduction_func "PCA" \
    --n_features 100 \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16

# All features
python -W ignore run_predict_traits.py data/outputs/biolog \
    --cross_validate \
    --random_state 42 \
    --n_cpus 16
