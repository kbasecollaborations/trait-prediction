#!/usr/bin/env bash

# Predict traits for metabolic phenotypes
echo "Predicting traits for metabolic phenotypes"
python predict_traits.py \
    "data/raw/metabolic_phenotypes_bacdive.tsv" \
    "data/raw/rast_features.tsv" \
    "data/outputs" \
    --feature_type "rast" \
    --random_state 42

# Predict traits for non-metabolic phenotypes
echo "Predicting traits for non-metabolic phenotypes"
python predict_traits.py \
    "data/raw/non_metabolic_phenotypes_bacdive.tsv" \
    "data/raw/rast_features.tsv" \
    "data/outputs" \
    --feature_type "rast" \
    --random_state 42
