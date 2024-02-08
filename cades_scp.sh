#!/usr/bin/env bash

scp -r trait_prediction notebooks tests cloud-cades:/datax/dkishore/trait-prediction/
scp -r *.py *.sh *.lock *.nix *.toml *.md LICENSE cloud-cades:/datax/dkishore/trait-prediction/
# scp -r data/processed cloud-cades:/datax/dkishore/trait-prediction/data/
scp -r data/processed/features_reduced cloud-cades:/datax/dkishore/trait-prediction/data/
