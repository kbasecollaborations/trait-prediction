"""Split-level ML utilities (``pipeline.split_ml``).

Covers column alignment (used directly by the manuscript), scoring,
feature-importance extraction, and the end-to-end ``train_and_evaluate`` path.
A small Random Forest keeps the end-to-end test fast.
"""

import numpy as np
import pandas as pd
import pytest

from trait_prediction.classifiers import make_classifier
from trait_prediction.pipeline.split_ml import (
    DEFAULT_SCORING,
    align_columns,
    get_feature_importances,
    get_scores,
    train_and_evaluate,
)


@pytest.fixture
def split_data():
    rng = np.random.default_rng(42)

    def block(prefix, n):
        return pd.DataFrame(
            rng.integers(0, 2, size=(n, 20)),
            index=[f"{prefix}_{i}" for i in range(n)],
            columns=[f"feature_{i}" for i in range(20)],
        )

    def labels(index):
        return pd.Series([0, 1] * (len(index) // 2), index=index)

    X_train, X_val, X_test = block("train", 60), block("val", 20), block("test", 20)
    return {
        "X_train": X_train,
        "y_train": labels(X_train.index),
        "X_val": X_val,
        "y_val": labels(X_val.index),
        "X_test": X_test,
        "y_test": labels(X_test.index),
    }


def test_align_columns_adds_missing_and_drops_extra():
    source = pd.DataFrame({"a": [1], "b": [2], "c": [3]}, index=["s"])
    target = pd.DataFrame({"a": [7], "b": [8], "z": [9]}, index=["t"])
    aligned = align_columns(source, target)

    assert list(aligned.columns) == ["a", "b", "c"]
    assert aligned["c"].tolist() == [0]  # missing column filled with 0
    assert "z" not in aligned.columns  # extra column dropped


def test_get_scores(split_data):
    model = make_classifier("rf", random_state=42, n_estimators=10)
    model.fit(split_data["X_train"], split_data["y_train"])
    scores = get_scores(model, split_data["X_test"], split_data["y_test"])

    assert set(DEFAULT_SCORING).issubset(scores)
    # most metrics are in [0, 1]; matthews_corrcoef is in [-1, 1]
    assert all(-1 <= v <= 1 for v in scores.values())


def test_get_feature_importances(split_data):
    model = make_classifier("rf", random_state=42, n_estimators=10)
    model.fit(split_data["X_train"], split_data["y_train"])
    importances = get_feature_importances(model, split_data["X_train"], n_features=5)

    assert isinstance(importances, pd.Series)
    assert len(importances) == 5
    assert importances.iloc[0] >= importances.iloc[-1]  # sorted descending


def test_train_and_evaluate(split_data):
    result = train_and_evaluate(
        **split_data,
        model_type="rf",
        random_state=42,
        n_estimators=10,
    )
    assert "balanced_accuracy" in result
    assert isinstance(result["features"], list)
    assert result["n_train"] == len(split_data["y_train"])
    assert result["n_test"] == len(split_data["y_test"])
