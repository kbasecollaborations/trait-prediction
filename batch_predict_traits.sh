#!/usr/bin/env bash

# This script is used to batch predict traits using the run_predict_traits.py script

## Commands

# All features
python -W ignore run_predict_traits.py data/outputs/biolog \
    --cross_validate \
    --n_cpus 4
# Reduction: NMF
python -W ignore run_predict_traits.py data/outputs/biolog_nmf_200 \
    --reduction_func "NMF" \
    --n_features 200 \
    --cross_validate \
    --n_cpus 4
# Reduction: PCA
python -W ignore run_predict_traits.py data/outputs/biolog_pca_200 \
    --reduction_func "PCA" \
    --n_features 200 \
    --cross_validate \
    --n_cpus 4
# Scoring: chi2
python -W ignore run_predict_traits.py data/outputs/biolog_chi2_1000 \
    --score_func "chi2" \
    --n_features 1000 \
    --cross_validate \
    --n_cpus 4
# Scoring: mutual_info_classif
python -W ignore run_predict_traits.py data/outputs/biolog_mutual_info_1000 \
    --score_func "mutual_info_classif" \
    --n_features 1000 \
    --cross_validate \
    --n_cpus 4
# Scoring: f_classif
python -W ignore run_predict_traits.py data/outputs/biolog_f_1000 \
    --score_func "f_classif" \
    --n_features 1000 \
    --cross_validate \
    --n_cpus 4
