# trait-prediction

A machine-learning framework for predicting microbial carbon-source utilisation
(and other binary traits) from genomic feature data.

This is the core Python library behind the manuscript *"From Spurious Shortcuts
to Mechanistic Specialists: Data Coherence and Applicability Domains in
Genome-Based Prediction of Microbial Carbon Utilisation."* It integrates binary
growth phenotypes across datasets and trains models under different train/test
splitting regimes to study cross-dataset generalisation. The data-processing and
experiment scripts used to produce the manuscript results live in the manuscript
repository; this repository contains only the reusable library.

## Installation

Uses [uv](https://docs.astral.sh/uv/) for dependency management (Python
3.12–3.13):

```bash
uv sync
```

## Quick start

```python
from trait_prediction.pipeline import run_pipeline

scores = run_pipeline(
    base_dir="<dir of split folders>",
    feature_file="<feature matrix>.parquet",
    split_types=["random", "out_of_clade"],
    model_type="catboost",
)
```

`run_pipeline` is a thin wrapper around `PipelineRunner` for quick experiments;
for full control construct a `Config`/`ConfigSet` and drive `TrainingPipeline`
directly. `configs/default.yaml` holds sensible defaults (CatBoost classifier,
χ² feature selection to 1000 features, variance/correlation filtering, 5-fold CV,
balanced-accuracy scoring) and `configs/config_set.yaml` sweeps multiple settings
via `ConfigSet`; the `run_script.py` entry point loads these with
[Hydra](https://hydra.cc/).

Run the test suite with `just test` (or `pytest`).

## Package layout

The library lives under `trait_prediction/`:

- **`main/`** — core data model: `DataSet`, `Feature`/`FeatureSet`,
  `Phenotype`/`PhenotypeSet`. Loads genomic features and binary phenotype
  labels and aligns them into trainable matrices. `feature_corr.py` provides a
  fast (Numba) correlation filter for redundant features.
- **`pipeline/`** — orchestration. `Config`/`ConfigSet` define a run;
  `PipelineRunner`/`run_pipeline` drive end-to-end training;
  `TrainingPipeline` handles preprocessing → feature selection → fit →
  evaluate; `splitters.py` (`RandomSplitter`, `InCladeSplitter`,
  `OutOfCladeSplitter`) implement the phylogeny-aware train/test regimes used
  to probe generalisation; `split_ml.py` holds scoring and feature-importance
  utilities.
- **`classifiers/`** — `make_classifier` factory (CatBoost, random forest,
  decision tree, RFE/RFECV) plus null-model baselines (`BernoulliClassifier`,
  `IdentityClassifier`, `NearestNeighborClassifier`) for comparison.
- **`training/`** — `Predictor` and training/CV data containers; sampling and
  Optuna-based hyperparameter optimisation.
- **`visualization/`** — SHAP-based feature-importance plots.

## License

See [LICENSE](LICENSE).
