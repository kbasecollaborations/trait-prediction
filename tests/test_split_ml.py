"""Tests for the split ML utilities module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from trait_prediction.classifiers import make_classifier
from trait_prediction.pipeline.split_ml import (
    DEFAULT_SCORING,
    align_columns,
    get_feature_importances,
    get_scores,
    train_and_evaluate,
)


@pytest.fixture
def sample_data():
    """Create sample train/val/test data for testing."""
    np.random.seed(42)
    n_train, n_val, n_test = 100, 20, 30
    n_features = 50

    # Create feature matrices
    X_train = pd.DataFrame(
        np.random.randint(0, 2, size=(n_train, n_features)),
        index=[f"train_{i}" for i in range(n_train)],
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    X_val = pd.DataFrame(
        np.random.randint(0, 2, size=(n_val, n_features)),
        index=[f"val_{i}" for i in range(n_val)],
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    X_test = pd.DataFrame(
        np.random.randint(0, 2, size=(n_test, n_features)),
        index=[f"test_{i}" for i in range(n_test)],
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Create labels (balanced classes)
    y_train = pd.Series(
        [0] * (n_train // 2) + [1] * (n_train // 2),
        index=X_train.index,
    )
    y_val = pd.Series(
        [0] * (n_val // 2) + [1] * (n_val // 2),
        index=X_val.index,
    )
    y_test = pd.Series(
        [0] * (n_test // 2) + [1] * (n_test // 2),
        index=X_test.index,
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


class TestGetScores:
    """Tests for the get_scores function."""

    def test_get_scores_basic(self, sample_data):
        """Test basic scoring functionality."""
        model = make_classifier("rf", random_state=42, n_estimators=10)
        model.fit(sample_data["X_train"], sample_data["y_train"])

        scores = get_scores(
            model,
            sample_data["X_test"],
            sample_data["y_test"],
            scoring=["accuracy", "balanced_accuracy"],
        )

        assert "accuracy" in scores
        assert "balanced_accuracy" in scores
        assert 0 <= scores["accuracy"] <= 1
        assert 0 <= scores["balanced_accuracy"] <= 1

    def test_get_scores_default_metrics(self, sample_data):
        """Test that default scoring metrics work."""
        model = make_classifier("rf", random_state=42, n_estimators=10)
        model.fit(sample_data["X_train"], sample_data["y_train"])

        scores = get_scores(model, sample_data["X_test"], sample_data["y_test"])

        for metric in DEFAULT_SCORING:
            assert metric in scores

    def test_get_scores_sensitivity_specificity(self, sample_data):
        """Test sensitivity and specificity metrics."""
        model = make_classifier("rf", random_state=42, n_estimators=10)
        model.fit(sample_data["X_train"], sample_data["y_train"])

        scores = get_scores(
            model,
            sample_data["X_test"],
            sample_data["y_test"],
            scoring=["sensitivity", "specificity"],
        )

        assert "sensitivity" in scores
        assert "specificity" in scores
        assert 0 <= scores["sensitivity"] <= 1
        assert 0 <= scores["specificity"] <= 1


class TestGetFeatureImportances:
    """Tests for the get_feature_importances function."""

    def test_feature_importances_rf(self, sample_data):
        """Test feature importances with Random Forest."""
        model = make_classifier("rf", random_state=42, n_estimators=10)
        model.fit(sample_data["X_train"], sample_data["y_train"])

        importances = get_feature_importances(
            model, sample_data["X_train"], n_features=10
        )

        assert isinstance(importances, pd.Series)
        assert len(importances) == 10
        # Check sorted in descending order
        assert importances.iloc[0] >= importances.iloc[-1]

    def test_feature_importances_all_features(self, sample_data):
        """Test returning all features."""
        model = make_classifier("rf", random_state=42, n_estimators=10)
        model.fit(sample_data["X_train"], sample_data["y_train"])

        importances = get_feature_importances(
            model, sample_data["X_train"], n_features=100
        )

        # Should return all features (50)
        assert len(importances) == 50


class TestAlignColumns:
    """Tests for the align_columns function."""

    def test_align_columns_matching(self):
        """Test alignment when columns already match."""
        X_source = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4], "c": [5, 6]},
            index=["s1", "s2"],
        )
        X_target = pd.DataFrame(
            {"a": [7, 8], "b": [9, 10], "c": [11, 12]},
            index=["t1", "t2"],
        )

        aligned = align_columns(X_source, X_target)

        assert list(aligned.columns) == list(X_source.columns)
        pd.testing.assert_frame_equal(aligned, X_target)

    def test_align_columns_missing(self):
        """Test alignment with missing columns in target."""
        X_source = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4], "c": [5, 6]},
            index=["s1", "s2"],
        )
        X_target = pd.DataFrame(
            {"a": [7, 8], "b": [9, 10]},  # Missing column 'c'
            index=["t1", "t2"],
        )

        aligned = align_columns(X_source, X_target)

        assert list(aligned.columns) == list(X_source.columns)
        assert aligned["c"].tolist() == [0, 0]  # Filled with 0

    def test_align_columns_extra(self):
        """Test alignment with extra columns in target."""
        X_source = pd.DataFrame(
            {"a": [1, 2], "b": [3, 4]},
            index=["s1", "s2"],
        )
        X_target = pd.DataFrame(
            {"a": [7, 8], "b": [9, 10], "c": [11, 12]},  # Extra column 'c'
            index=["t1", "t2"],
        )

        aligned = align_columns(X_source, X_target)

        assert list(aligned.columns) == list(X_source.columns)
        assert "c" not in aligned.columns


class TestTrainAndEvaluate:
    """Tests for the train_and_evaluate function."""

    def test_train_and_evaluate_rf(self, sample_data):
        """Test train and evaluate with Random Forest."""
        result = train_and_evaluate(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            X_test=sample_data["X_test"],
            y_test=sample_data["y_test"],
            model_type="rf",
            random_state=42,
            n_estimators=10,  # Small for speed
        )

        # Check scores are present
        assert "accuracy" in result
        assert "balanced_accuracy" in result

        # Check features are present
        assert "features" in result
        assert isinstance(result["features"], list)

        # Check sample counts
        assert result["n_train"] == len(sample_data["y_train"])
        assert result["n_val"] == len(sample_data["y_val"])
        assert result["n_test"] == len(sample_data["y_test"])

    def test_train_and_evaluate_catboost_noeval(self, sample_data):
        """Test train and evaluate with CatBoost (no eval)."""
        result = train_and_evaluate(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            X_test=sample_data["X_test"],
            y_test=sample_data["y_test"],
            model_type="catboost_noeval",
            random_state=42,
            iterations=10,  # Small for speed
        )

        assert "accuracy" in result
        assert "features" in result

    def test_train_and_evaluate_custom_scoring(self, sample_data):
        """Test train and evaluate with custom scoring metrics."""
        result = train_and_evaluate(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            X_test=sample_data["X_test"],
            y_test=sample_data["y_test"],
            model_type="rf",
            scoring=["accuracy", "f1"],
            random_state=42,
            n_estimators=10,
        )

        assert "accuracy" in result
        assert "f1" in result
        # Should not have metrics not in scoring
        assert "matthews_corrcoef" not in result
