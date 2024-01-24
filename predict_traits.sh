#!/usr/bin/env bash

function predict_traits {
    python predict_traits.py \
        "$1" \
        "$2" \
        "$3" \
        --feature_type "generic" \
        --random_state 42 \
        --overwrite
}

# Predict traits for biolog data
echo "Predicting traits for pmi biolog"
predict_traits "data/processed/biolog/phenotypes/pmi_phenotypes.tsv" \
    "data/processed/biolog/features/rast/rast_pmi_features.tsv" \
    "data/outputs/biolog/rast"
predict_traits "data/processed/biolog/phenotypes/pmi_phenotypes.tsv" \
    "data/processed/biolog/features/kofam/kofam_pmi_features.tsv" \
    "data/outputs/biolog/kofam"

echo "Predicting traits for ch biolog"
predict_traits "data/processed/biolog/phenotypes/ch_phenotypes.tsv" \
    "data/processed/biolog/features/rast/rast_ch_features.tsv" \
    "data/outputs/biolog/rast"
predict_traits "data/processed/biolog/phenotypes/ch_phenotypes.tsv" \
    "data/processed/biolog/features/kofam/kofam_ch_features.tsv" \
    "data/outputs/biolog/kofam"

echo "Predicting traits for leaf biolog"
predict_traits "data/processed/biolog/phenotypes/leaf_phenotypes.tsv" \
    "data/processed/biolog/features/rast/rast_leaf_features.tsv" \
    "data/outputs/biolog/rast"
predict_traits "data/processed/biolog/phenotypes/leaf_phenotypes.tsv" \
    "data/processed/biolog/features/kofam/kofam_leaf_features.tsv" \
    "data/outputs/biolog/kofam"

echo "Predicing using uniref30 features"
predict_traits "data/processed/biolog/phenotypes/pmi_phenotypes.tsv" \
    "data/processed/biolog/features/uniref30/uniref30_pmi_features.tsv" \
    "data/outputs/biolog/uniref30"
predict_traits "data/processed/biolog/phenotypes/ch_phenotypes.tsv" \
    "data/processed/biolog/features/uniref30/uniref30_ch_features.tsv" \
    "data/outputs/biolog/uniref30"
predict_traits "data/processed/biolog/phenotypes/leaf_phenotypes.tsv" \
    "data/processed/biolog/features/uniref30/uniref30_leaf_features.tsv" \
    "data/outputs/biolog/uniref30"

# # Predict traits for metabolic phenotypes
# echo "Predicting traits for metabolic phenotypes"
# predict_traits \
#     "data/raw/metabolic_phenotypes_bacdive.tsv" \
#     "data/raw/rast_features.tsv" \
#     "data/outputs" \
#     --feature_type "rast" \
#     --random_state 42 \
#     --cross_validate \
#     --overwrite

# # Predict traits for non-metabolic phenotypes
# echo "Predicting traits for non-metabolic phenotypes"
# predict_traits \
#     "data/raw/non_metabolic_phenotypes_bacdive.tsv" \
#     "data/raw/rast_features.tsv" \
#     "data/outputs" \
#     --feature_type "rast" \
#     --random_state 42 \
#     --cross_validate \
#     --overwrite
