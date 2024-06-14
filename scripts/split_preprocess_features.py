#!/usr/bin/env python
# coding: utf-8

# # Notebook to prepare processed data for training the phenotype prediction classifiers
#
# Current datasets include:
# 1. Literature
# 2. AT Leaf
# 3. PMI
#
# Current features include:
# - cluster30
# - cluster50
# - cluster70
# - cluster90
# - eggnog_kegg
# - eggnog_seed
# - kofam_modules
# - kofam
# - rast
# - uniprot_trembl
# - uniref30
# - uniref50
# - uniref70
# - uniref90

# In[ ]:


import argparse
import gzip
import json
from pathlib import Path
from typing import Literal

import pandas as pd

from trait_prediction.main import Feature, FeatureIndex, FeatureInput

# In[ ]:


# In[ ]:


# In[ ]:


# Parameters
INPUT_FOLDER = Path("../data/raw/features/biolog/")
OUTPUT_FOLDER = Path("../data/processed/features_reduced")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
VARIANCE_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95


# ## Create the FeatureInputs

# In[ ]:


feature_inputs: list[FeatureInput] = []
for feature_file in Path(INPUT_FOLDER).glob("*.tsv"):
    name = feature_file.stem
    if name == "kofam_modules":
        ftype = "float"
        dtype = "float64"
    else:
        # NOTE: We are binarizing the counts here
        ftype = "binary"
        dtype = "uint8"
    findex = FeatureIndex(
        name=name,
        ftype=ftype,
        dtype=dtype,
    )
    feature_input = FeatureInput(
        path=feature_file,
        findex=findex,
        index_format_func=lambda x: x.strip()
        .split("?")[-1]
        .removesuffix(".RAST")
        .removesuffix(".fna"),
    )
    feature_inputs.append(feature_input)


# ## Read the raw phenotype data

# In[ ]:


PHENOTYPE_FOLDER = Path("../data/raw/phenotypes")
index_format_func = (
    lambda x: x.strip()
    .split("?")[-1]
    .removesuffix(".RAST")
    .removesuffix(".fna")
    .removeprefix("g")
)
lit_phenotype = pd.read_csv(
    PHENOTYPE_FOLDER / "lit_phenotypes.tsv", sep="\t", index_col=0
)
lit_index = pd.Series(lit_phenotype.index)
lit_phenotype.index = lit_index.apply(index_format_func)
lit_phenotype.index.name = "genomeID"
atleaf_phenotype = pd.read_csv(
    PHENOTYPE_FOLDER / "atleaf_phenotypes.tsv", sep="\t", index_col=0
)
atleaf_index = pd.Series(atleaf_phenotype.index)
atleaf_phenotype.index.name = "genomeID"
atleaf_phenotype.index = atleaf_index.apply(index_format_func)
pmi_phenotype = pd.read_csv(
    PHENOTYPE_FOLDER / "pmi_phenotypes.tsv", sep="\t", index_col=0
)
pmi_index = pd.Series(pmi_phenotype.index)
pmi_phenotype.index = pmi_index.apply(index_format_func)
pmi_phenotype.index.name = "genomeID"


# ## Preprocess the feature data

# In[ ]:


from tqdm import tqdm

# In[ ]:


datasets = {
    "lit": lit_phenotype,
    "atleaf": atleaf_phenotype,
    "pmi": pmi_phenotype,
}
pbar = tqdm(feature_inputs)
for feature_input in pbar:
    feature = Feature.read_data(feature_input)
    original_feature_data = feature.feature_data
    for dataset_name, dataset in datasets.items():
        name = feature_input.findex.name
        output_dir = OUTPUT_FOLDER / f"{dataset_name}/{feature_input.findex.name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        pbar.set_description(
            f"Processing {feature_input.findex.name} for {dataset_name}"
        )
        common_index = original_feature_data.index.intersection(dataset.index)
        feature_data = original_feature_data.loc[common_index, :]
        feature_data, low_var_features_list = Feature.remove_features_with_low_variance(
            feature_data, VARIANCE_THRESHOLD
        )
        feature_data, high_corr_features_dict = (
            Feature.remove_features_with_high_correlation(
                feature_data, CORRELATION_THRESHOLD, parallel=True
            )
        )
        # Save the data
        output_file = output_dir / f"{name}.tsv"
        low_var_file = output_dir / f"{name}_low_var_features.txt"
        corr_file = output_dir / f"{name}_corr_features.json.gz"
        feature_data.to_csv(output_file, sep="\t", index=True)
        with open(low_var_file, "w") as fid:
            fid.write("\n".join(low_var_features_list))
        with gzip.open(corr_file, "wt") as gzfile:
            json.dump(high_corr_features_dict, gzfile)


# In[ ]:
