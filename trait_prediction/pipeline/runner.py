"""High-level pipeline runner for batch ML experiments.

This module provides a configuration-driven interface for running ML experiments
across combinations of datasets, features, phenotypes, and split types.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from trait_prediction.classifiers import make_classifier
from trait_prediction.pipeline.split_ml import (
    DEFAULT_SCORING,
    align_columns,
    get_feature_importances,
    get_scores,
    load_single_split,
)


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline execution.

    Parameters
    ----------
    model_type
        Type of classifier to use ("catboost", "catboost_noeval", "rf", "dt").
    scoring
        List of scoring metrics to evaluate.
    random_state
        Random seed for reproducibility.
    n_top_features
        Number of top features to return in results.
    model_kwargs
        Additional keyword arguments passed to make_classifier.

    Examples
    --------
    >>> config = PipelineConfig(
    ...     model_type="catboost",
    ...     scoring=["balanced_accuracy", "f1"],
    ...     random_state=42,
    ... )
    """

    model_type: str = "catboost"
    scoring: list[str] = field(default_factory=lambda: DEFAULT_SCORING.copy())
    random_state: int = 42
    n_top_features: int = 10
    model_kwargs: dict[str, Any] = field(default_factory=dict)


class PipelineRunner:
    """High-level runner for ML experiments on pre-computed splits.

    This class provides a simple interface for running ML experiments across
    multiple split types, phenotypes, and feature sets. It handles data loading,
    model training, evaluation, and result collection.

    Parameters
    ----------
    config
        Pipeline configuration.

    Examples
    --------
    >>> config = PipelineConfig(model_type="catboost")
    >>> runner = PipelineRunner(config)
    >>> results = runner.run(
    ...     base_dir=Path("data/splits"),
    ...     feature_file=Path("data/features.tsv"),
    ...     split_types=["random_split"],
    ... )
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    def run(
        self,
        base_dir: Path,
        feature_file: Path,
        split_types: Sequence[str] | None = None,
        phenotypes: Sequence[str] | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Run ML experiments on all matching splits.

        Parameters
        ----------
        base_dir
            Base directory containing split folders.
        feature_file
            Path to feature file (TSV format).
        split_types
            Split types to process. If None, processes all available.
        phenotypes
            Phenotypes to include. If None, includes all available.
        verbose
            Whether to show progress bars.

        Returns
        -------
        pd.DataFrame
            Results DataFrame with columns for each metric plus metadata:
            - split_type: Type of split (random_split, dataset_split, etc.)
            - split_key: Unique identifier for the split
            - phenotype: Phenotype name
            - features: List of top feature names
            - n_train, n_val, n_test: Sample counts
        """
        if split_types is None:
            split_types = ["random_split", "dataset_split", "phylo_ooc", "phylo_ic"]

        # Load feature data
        if verbose:
            print(f"Loading feature data from {feature_file}")
        feature_data = pd.read_csv(
            feature_file, sep="\t", index_col=0, dtype={"genomeID": str}
        )
        if verbose:
            print(f"Feature data shape: {feature_data.shape}")

        results: list[dict[str, Any]] = []

        for split_type in split_types:
            split_dir = self._get_split_dir(base_dir, split_type)
            if split_dir is None or not split_dir.exists():
                if verbose:
                    print(f"Skipping {split_type}: directory not found")
                continue

            split_results = self._run_split_type(
                split_dir=split_dir,
                split_type=split_type,
                feature_data=feature_data,
                phenotypes=phenotypes,
                verbose=verbose,
            )
            results.extend(split_results)

        return pd.DataFrame(results)

    def run_single(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict[str, Any]:
        """Run ML on a single pre-loaded split.

        Parameters
        ----------
        X_train, y_train
            Training data.
        X_val, y_val
            Validation data (for early stopping).
        X_test, y_test
            Test data for evaluation.

        Returns
        -------
        dict[str, Any]
            Results dictionary with scores and feature importances.
        """
        model = make_classifier(
            self.config.model_type,
            random_state=self.config.random_state,
            **self.config.model_kwargs,
        )

        # Align columns
        X_val_aligned = align_columns(X_train, X_val)
        X_test_aligned = align_columns(X_train, X_test)

        # Train
        if self.config.model_type == "catboost":
            model.fit(
                X_train,
                y_train,
                eval_set=(X_val_aligned, y_val),
                use_best_model=True,
                verbose=False,
            )
        elif self.config.model_type == "catboost_noeval":
            model.fit(X_train, y_train, verbose=False)
        else:
            model.fit(X_train, y_train)

        # Evaluate
        scores = get_scores(model, X_test_aligned, y_test, self.config.scoring)

        # Feature importances
        importances = get_feature_importances(
            model, X_train, n_features=self.config.n_top_features
        )
        scores["features"] = importances.index.tolist()
        scores["n_train"] = len(y_train)
        scores["n_val"] = len(y_val)
        scores["n_test"] = len(y_test)

        return scores

    def _get_split_dir(self, base_dir: Path, split_type: str) -> Path | None:
        """Get the directory for a split type."""
        if split_type == "random_split":
            return base_dir / "random_split"
        elif split_type == "dataset_split":
            return base_dir / "dataset_split"
        elif split_type == "phylo_ooc":
            return base_dir / "phylogeny_split"
        elif split_type == "phylo_ic":
            return base_dir / "phylogeny_split"
        return None

    def _run_split_type(
        self,
        split_dir: Path,
        split_type: str,
        feature_data: pd.DataFrame,
        phenotypes: Sequence[str] | None,
        verbose: bool,
    ) -> list[dict[str, Any]]:
        """Run ML on all splits of a given type."""
        results: list[dict[str, Any]] = []

        phenotype_dirs = list(split_dir.iterdir())
        iterator = (
            tqdm(phenotype_dirs, desc=f"Processing {split_type}")
            if verbose
            else phenotype_dirs
        )

        for phenotype_dir in iterator:
            if not phenotype_dir.is_dir():
                continue

            phenotype_name = phenotype_dir.name

            # Filter by phenotype if specified
            if phenotypes is not None and phenotype_name not in phenotypes:
                continue

            # Handle phylogenetic split subdirectories
            if split_type == "phylo_ooc":
                sub_dir = phenotype_dir / "out-of-clade"
                if not sub_dir.exists():
                    continue
                split_dirs = list(sub_dir.iterdir())
            elif split_type == "phylo_ic":
                sub_dir = phenotype_dir / "in-clade"
                if not sub_dir.exists():
                    continue
                split_dirs = list(sub_dir.iterdir())
            else:
                split_dirs = list(phenotype_dir.iterdir())

            for split_subdir in split_dirs:
                if not split_subdir.is_dir():
                    continue

                try:
                    data = load_single_split(split_subdir, feature_data)

                    # Skip if insufficient data
                    if len(data["y_train"]) < 10 or len(data["y_test"]) < 5:
                        continue

                    # Skip if only one class
                    if len(data["y_train"].unique()) < 2:
                        continue

                    result = self.run_single(
                        X_train=data["X_train"],
                        y_train=data["y_train"],
                        X_val=data["X_val"],
                        y_val=data["y_val"],
                        X_test=data["X_test"],
                        y_test=data["y_test"],
                    )

                    # Add metadata
                    result["split_type"] = split_type
                    result["split_key"] = split_subdir.name
                    result["phenotype"] = phenotype_name

                    results.append(result)

                except Exception as e:
                    if verbose:
                        print(f"Error processing {split_subdir}: {e}")
                    continue

        return results


def run_pipeline(
    base_dir: Path | str,
    feature_file: Path | str,
    split_types: Sequence[str] | None = None,
    phenotypes: Sequence[str] | None = None,
    model_type: str = "catboost",
    scoring: list[str] | None = None,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Convenience function to run ML pipeline with minimal setup.

    This is a simple wrapper around PipelineRunner for quick experiments.

    Parameters
    ----------
    base_dir
        Base directory containing split folders.
    feature_file
        Path to feature file.
    split_types
        Split types to process.
    phenotypes
        Phenotypes to include.
    model_type
        Classifier type.
    scoring
        Scoring metrics.
    random_state
        Random seed.
    verbose
        Show progress.

    Returns
    -------
    pd.DataFrame
        Results DataFrame.

    Examples
    --------
    >>> results = run_pipeline(
    ...     base_dir="data/splits",
    ...     feature_file="data/features.tsv",
    ...     split_types=["random_split"],
    ...     phenotypes=["Alanine", "Glucose"],
    ...     model_type="catboost",
    ... )
    """
    config = PipelineConfig(
        model_type=model_type,
        scoring=scoring or DEFAULT_SCORING.copy(),
        random_state=random_state,
    )

    runner = PipelineRunner(config)
    return runner.run(
        base_dir=Path(base_dir),
        feature_file=Path(feature_file),
        split_types=split_types,
        phenotypes=phenotypes,
        verbose=verbose,
    )
