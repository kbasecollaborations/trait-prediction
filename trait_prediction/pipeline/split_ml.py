"""Machine learning utilities for train-val-test split workflows.

This module provides utilities for running ML pipelines on pre-generated
train/val/test splits, with support for early stopping and proper column
alignment between splits.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFE
from sklearn.metrics import get_scorer
from tqdm import tqdm

from trait_prediction.classifiers import make_classifier

# Default scoring metrics for classification
DEFAULT_SCORING: list[str] = [
    "accuracy",
    "balanced_accuracy",
    "matthews_corrcoef",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]


def get_scores(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: list[str] | None = None,
) -> dict[str, float]:
    """Calculate multiple scoring metrics for a trained model.

    Parameters
    ----------
    model
        Trained scikit-learn compatible model.
    X
        Feature matrix.
    y
        True labels.
    scoring
        List of scoring metric names. Supports all sklearn scorers plus
        "sensitivity" and "specificity" as aliases for recall with
        pos_label=1 and pos_label=0 respectively.
        If None, uses DEFAULT_SCORING.

    Returns
    -------
    dict[str, float]
        Dictionary mapping metric names to scores.
    """
    if scoring is None:
        scoring = DEFAULT_SCORING

    results: dict[str, float] = {}
    for metric_name in scoring:
        if metric_name == "sensitivity":
            scorer = get_scorer("recall")
            score = scorer(model, X, y, pos_label=1)
        elif metric_name == "specificity":
            scorer = get_scorer("recall")
            score = scorer(model, X, y, pos_label=0)
        else:
            scorer = get_scorer(metric_name)
            score = scorer(model, X, y)
        results[metric_name] = score

    return results


def get_feature_importances(
    model: BaseEstimator,
    X: pd.DataFrame,
    feature_names: list[str] | None = None,
    n_features: int = 10,
) -> pd.Series:
    """Extract top feature importances from a trained model.

    Parameters
    ----------
    model
        Trained model with `feature_importances_` attribute
        (e.g., RandomForest, CatBoost, DecisionTree) or RFE wrapper.
    X
        Feature matrix used for training (used to get column names).
    feature_names
        Explicit list of feature names. If None, uses X.columns.
        Must be None for RFE models.
    n_features
        Number of top features to return.

    Returns
    -------
    pd.Series
        Series of feature importances, indexed by feature name,
        sorted in descending order (most important first).

    Raises
    ------
    ValueError
        If feature_names is provided for an RFE model.
    """
    if isinstance(model, RFE):
        estimator = model.estimator_
        if feature_names is not None:
            raise ValueError("feature_names must be None for RFE models")
        feature_names = list(model.get_feature_names_out())
    else:
        estimator = model

    importances = estimator.feature_importances_

    if feature_names is None:
        feature_names = list(X.columns)

    importance_series = pd.Series(importances, index=feature_names)
    importance_series.sort_values(ascending=False, inplace=True)

    return importance_series.head(n_features)


def align_columns(
    X_source: pd.DataFrame,
    X_target: pd.DataFrame,
    fill_value: int | float = 0,
) -> pd.DataFrame:
    """Align columns of target DataFrame to match source DataFrame.

    Adds missing columns filled with fill_value and reorders columns
    to match the source DataFrame.

    Parameters
    ----------
    X_source
        Reference DataFrame whose columns to match.
    X_target
        DataFrame to align.
    fill_value
        Value to fill missing columns with.

    Returns
    -------
    pd.DataFrame
        Target DataFrame with columns aligned to source.
    """
    X_aligned = X_target.copy()

    missing_cols = X_source.columns.difference(X_aligned.columns)
    if len(missing_cols) > 0:
        missing_df = pd.DataFrame(fill_value, index=X_aligned.index, columns=missing_cols)
        X_aligned = pd.concat([X_aligned, missing_df], axis=1)

    return X_aligned[X_source.columns]


def load_single_split(
    folder_path: Path,
    feature_data: pd.DataFrame,
    index_col: str = "genomeID",
) -> dict[str, pd.DataFrame | pd.Series]:
    """Load train/val/test data from a single split folder.

    Split folders contain only phenotype labels (y_train.tsv, y_val.tsv, y_test.tsv).
    Feature data is provided separately and aligned with split indices.

    Parameters
    ----------
    folder_path
        Path to folder containing split files.
    feature_data
        Feature matrix with samples as rows and features as columns.
    index_col
        Name of the index column in split files.

    Returns
    -------
    dict[str, pd.DataFrame | pd.Series]
        Dictionary with keys: X_train, y_train, X_val, y_val, X_test, y_test.
    """
    data: dict[str, pd.DataFrame | pd.Series] = {}

    for split_name in ["train", "val", "test"]:
        # Load labels
        y_file = folder_path / f"y_{split_name}.tsv"
        y_data = pd.read_csv(y_file, sep="\t", index_col=0, dtype={index_col: str})

        # Convert to Series if single column
        if isinstance(y_data, pd.DataFrame) and y_data.shape[1] == 1:
            y_data = y_data.iloc[:, 0]

        # Get indices present in both split and feature data
        valid_indices = y_data.index.intersection(feature_data.index)

        # Create aligned X and y
        X_data = feature_data.loc[valid_indices, :]
        y_data = y_data.loc[valid_indices]

        data[f"X_{split_name}"] = X_data
        data[f"y_{split_name}"] = y_data

    return data


def load_splits(
    base_dir: Path,
    feature_file: Path,
    split_types: list[str] | None = None,
    index_col: str = "genomeID",
    verbose: bool = True,
) -> dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]:
    """Load all train/val/test splits from a directory structure.

    Supports multiple split types:
    - random_split: Random train/val/test splits
    - dataset_split: Leave-one-dataset-out splits
    - phylo_ooc: Phylogenetic out-of-clade splits
    - phylo_ic: Phylogenetic in-clade splits

    Parameters
    ----------
    base_dir
        Base directory containing split folders.
    feature_file
        Path to feature file (TSV with samples as rows).
    split_types
        List of split types to load. If None, loads all available.
    index_col
        Name of the index column in files.
    verbose
        Whether to print progress information.

    Returns
    -------
    dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]]
        Nested dictionary: {split_type: {split_key: {X_train, y_train, ...}}}

    Examples
    --------
    >>> splits = load_splits(
    ...     base_dir=Path("data/splits"),
    ...     feature_file=Path("data/features.tsv"),
    ...     split_types=["random_split"],
    ... )
    >>> X_train = splits["random_split"]["Alanine_0"]["X_train"]
    """
    if split_types is None:
        split_types = ["random_split", "dataset_split", "phylo_ooc", "phylo_ic"]

    # Load feature data once
    if verbose:
        print(f"Loading feature data from {feature_file}")
    feature_data = pd.read_csv(
        feature_file, sep="\t", index_col=0, dtype={index_col: str}
    )
    if verbose:
        print(f"Feature data shape: {feature_data.shape}")

    result: dict[str, dict[str, dict[str, pd.DataFrame | pd.Series]]] = {}

    # Load random splits
    if "random_split" in split_types:
        random_dir = base_dir / "random_split"
        if random_dir.exists():
            random_data: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
            phenotype_dirs = list(random_dir.iterdir())
            iterator = tqdm(phenotype_dirs, desc="Loading random splits") if verbose else phenotype_dirs

            for phenotype_dir in iterator:
                if not phenotype_dir.is_dir():
                    continue
                for repeat_dir in phenotype_dir.iterdir():
                    if not repeat_dir.is_dir():
                        continue
                    key = f"{phenotype_dir.name}_{repeat_dir.name}"
                    random_data[key] = load_single_split(repeat_dir, feature_data, index_col)

            result["random_split"] = random_data

    # Load dataset splits
    if "dataset_split" in split_types:
        dataset_dir = base_dir / "dataset_split"
        if dataset_dir.exists():
            dataset_data: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
            phenotype_dirs = list(dataset_dir.iterdir())
            iterator = tqdm(phenotype_dirs, desc="Loading dataset splits") if verbose else phenotype_dirs

            for phenotype_dir in iterator:
                if not phenotype_dir.is_dir():
                    continue
                for split_dir in phenotype_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_dir.name}_{split_dir.name}"
                    dataset_data[key] = load_single_split(split_dir, feature_data, index_col)

            result["dataset_split"] = dataset_data

    # Load phylogenetic out-of-clade splits
    if "phylo_ooc" in split_types:
        phylo_dir = base_dir / "phylogeny_split"
        if phylo_dir.exists():
            ooc_data: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
            phenotype_dirs = list(phylo_dir.iterdir())
            iterator = tqdm(phenotype_dirs, desc="Loading phylo OOC splits") if verbose else phenotype_dirs

            for phenotype_dir in iterator:
                if not phenotype_dir.is_dir():
                    continue
                ooc_dir = phenotype_dir / "out-of-clade"
                if not ooc_dir.exists():
                    continue
                for split_dir in ooc_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_dir.name}_ooc_{split_dir.name}"
                    ooc_data[key] = load_single_split(split_dir, feature_data, index_col)

            result["phylo_ooc"] = ooc_data

    # Load phylogenetic in-clade splits
    if "phylo_ic" in split_types:
        phylo_dir = base_dir / "phylogeny_split"
        if phylo_dir.exists():
            ic_data: dict[str, dict[str, pd.DataFrame | pd.Series]] = {}
            phenotype_dirs = list(phylo_dir.iterdir())
            iterator = tqdm(phenotype_dirs, desc="Loading phylo IC splits") if verbose else phenotype_dirs

            for phenotype_dir in iterator:
                if not phenotype_dir.is_dir():
                    continue
                ic_dir = phenotype_dir / "in-clade"
                if not ic_dir.exists():
                    continue
                for split_dir in ic_dir.iterdir():
                    if not split_dir.is_dir():
                        continue
                    key = f"{phenotype_dir.name}_ic_{split_dir.name}"
                    ic_data[key] = load_single_split(split_dir, feature_data, index_col)

            result["phylo_ic"] = ic_data

    return result


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "catboost",
    scoring: list[str] | None = None,
    n_top_features: int = 10,
    **model_kwargs: Any,
) -> dict[str, Any]:
    """Train a model and evaluate on test set with validation-based early stopping.

    This function:
    1. Aligns columns between train/val/test sets
    2. Trains the model (with early stopping for CatBoost)
    3. Evaluates on test set
    4. Extracts feature importances

    Parameters
    ----------
    X_train
        Training feature matrix.
    y_train
        Training labels.
    X_val
        Validation feature matrix (used for early stopping).
    y_val
        Validation labels.
    X_test
        Test feature matrix.
    y_test
        Test labels.
    model_type
        Type of classifier ("catboost", "catboost_noeval", "rf", "dt", etc.).
    scoring
        List of scoring metrics. If None, uses DEFAULT_SCORING.
    n_top_features
        Number of top features to return.
    **model_kwargs
        Additional arguments passed to make_classifier.

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - All scoring metric values
        - "features": list of top feature names
        - "n_train": number of training samples
        - "n_val": number of validation samples
        - "n_test": number of test samples
    """
    if scoring is None:
        scoring = DEFAULT_SCORING

    model = make_classifier(model_type, **model_kwargs)

    # Align columns
    X_val_aligned = align_columns(X_train, X_val)
    X_test_aligned = align_columns(X_train, X_test)

    # Train model
    if model_type == "catboost":
        # CatBoost with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val_aligned, y_val),
            use_best_model=True,
            verbose=False,
        )
    elif model_type == "catboost_noeval":
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)

    # Evaluate
    scores = get_scores(model, X_test_aligned, y_test, scoring)

    # Get feature importances
    feature_importances = get_feature_importances(model, X_train, n_features=n_top_features)
    scores["features"] = feature_importances.index.tolist()

    # Add sample counts
    scores["n_train"] = len(y_train)
    scores["n_val"] = len(y_val)
    scores["n_test"] = len(y_test)

    return scores
