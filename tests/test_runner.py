"""Tests for the pipeline runner module."""

import numpy as np
import pandas as pd
import pytest

from trait_prediction.pipeline.runner import (
    PipelineConfig,
    PipelineRunner,
)
from trait_prediction.pipeline.split_ml import DEFAULT_SCORING


@pytest.fixture
def sample_data():
    """Create sample train/val/test data for testing."""
    np.random.seed(42)
    n_train, n_val, n_test = 100, 20, 30
    n_features = 50

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


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.model_type == "catboost"
        assert config.scoring == DEFAULT_SCORING
        assert config.random_state == 42
        assert config.n_top_features == 10
        assert config.model_kwargs == {}

    def test_custom_config(self):
        """Test custom configuration values."""
        config = PipelineConfig(
            model_type="rf",
            scoring=["accuracy", "f1"],
            random_state=123,
            n_top_features=20,
            model_kwargs={"n_estimators": 500},
        )

        assert config.model_type == "rf"
        assert config.scoring == ["accuracy", "f1"]
        assert config.random_state == 123
        assert config.n_top_features == 20
        assert config.model_kwargs == {"n_estimators": 500}


class TestPipelineRunner:
    """Tests for PipelineRunner class."""

    def test_runner_init_default(self):
        """Test runner initialization with default config."""
        runner = PipelineRunner()

        assert runner.config.model_type == "catboost"
        assert runner.config.random_state == 42

    def test_runner_init_custom_config(self):
        """Test runner initialization with custom config."""
        config = PipelineConfig(model_type="rf", random_state=123)
        runner = PipelineRunner(config)

        assert runner.config.model_type == "rf"
        assert runner.config.random_state == 123

    def test_run_single_rf(self, sample_data):
        """Test running on a single split with Random Forest."""
        config = PipelineConfig(
            model_type="rf",
            scoring=["accuracy", "balanced_accuracy"],
            model_kwargs={"n_estimators": 10},
        )
        runner = PipelineRunner(config)

        result = runner.run_single(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            X_test=sample_data["X_test"],
            y_test=sample_data["y_test"],
        )

        # Check scores
        assert "accuracy" in result
        assert "balanced_accuracy" in result
        assert 0 <= result["accuracy"] <= 1

        # Check features
        assert "features" in result
        assert isinstance(result["features"], list)

        # Check sample counts
        assert result["n_train"] == len(sample_data["y_train"])
        assert result["n_val"] == len(sample_data["y_val"])
        assert result["n_test"] == len(sample_data["y_test"])

    def test_run_single_catboost_noeval(self, sample_data):
        """Test running on a single split with CatBoost (no eval)."""
        config = PipelineConfig(
            model_type="catboost_noeval",
            scoring=["accuracy"],
            model_kwargs={"iterations": 10},
        )
        runner = PipelineRunner(config)

        result = runner.run_single(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            X_test=sample_data["X_test"],
            y_test=sample_data["y_test"],
        )

        assert "accuracy" in result
        assert "features" in result

    def test_run_single_with_column_mismatch(self, sample_data):
        """Test that column alignment works correctly."""
        config = PipelineConfig(
            model_type="rf",
            model_kwargs={"n_estimators": 10},
        )
        runner = PipelineRunner(config)

        # Remove some columns from test data
        X_test_missing = sample_data["X_test"].drop(
            columns=["feature_0", "feature_1"]
        )

        result = runner.run_single(
            X_train=sample_data["X_train"],
            y_train=sample_data["y_train"],
            X_val=sample_data["X_val"],
            y_val=sample_data["y_val"],
            X_test=X_test_missing,
            y_test=sample_data["y_test"],
        )

        # Should still work with column alignment
        assert "accuracy" in result
