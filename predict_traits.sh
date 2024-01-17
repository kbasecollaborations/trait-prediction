#!/usr/bin/env bash

# Predict traits for biolog data
echo "Predicting traits for ch biolog"
python predict_traits.py \
    "data/processed/biolog/phenotypes/ch_phenotypes.tsv" \
    "data/processed/biolog/features/rast/rast_ch_features.tsv" \
    "data/outputs/biolog" \
    --feature_type "generic" \
    --limit 10 \
    --random_state 42 \
    --overwrite

# # Predict traits for metabolic phenotypes
# echo "Predicting traits for metabolic phenotypes"
# python predict_traits.py \
#     "data/raw/metabolic_phenotypes_bacdive.tsv" \
#     "data/raw/rast_features.tsv" \
#     "data/outputs" \
#     --feature_type "rast" \
#     --random_state 42 \
#     --cross_validate \
#     --overwrite

# # Predict traits for non-metabolic phenotypes
# echo "Predicting traits for non-metabolic phenotypes"
# python predict_traits.py \
#     "data/raw/non_metabolic_phenotypes_bacdive.tsv" \
#     "data/raw/rast_features.tsv" \
#     "data/outputs" \
#     --feature_type "rast" \
#     --random_state 42 \
#     --cross_validate \
#     --overwrite
