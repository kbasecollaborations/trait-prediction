"""Tests for the classifier factory module."""

import pytest
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.tree import DecisionTreeClassifier

from trait_prediction.classifiers import (
    CATBOOST_DEFAULTS,
    CATBOOST_NOEVAL_DEFAULTS,
    DT_DEFAULTS,
    RF_DEFAULTS,
    make_classifier,
)


class TestMakeClassifier:
    """Tests for the make_classifier factory function."""

    def test_make_classifier_rf(self):
        """Test creating a Random Forest classifier."""
        model = make_classifier("rf", random_state=42)
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == 42
        assert model.n_estimators == RF_DEFAULTS["n_estimators"]
        assert model.max_features == RF_DEFAULTS["max_features"]

    def test_make_classifier_rf_custom_params(self):
        """Test creating RF with custom parameters."""
        model = make_classifier("rf", random_state=42, n_estimators=500, max_depth=10)
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 500
        assert model.max_depth == 10

    def test_make_classifier_dt(self):
        """Test creating a Decision Tree classifier."""
        model = make_classifier("dt", random_state=42)
        assert isinstance(model, DecisionTreeClassifier)
        assert model.random_state == 42
        assert model.criterion == DT_DEFAULTS["criterion"]

    def test_make_classifier_catboost(self):
        """Test creating a CatBoost classifier with early stopping."""
        model = make_classifier("catboost", random_state=42)
        assert isinstance(model, CatBoostClassifier)
        params = model.get_params()
        assert params["random_state"] == 42
        assert params["iterations"] == CATBOOST_DEFAULTS["iterations"]
        assert params["depth"] == CATBOOST_DEFAULTS["depth"]
        assert params["use_best_model"] == CATBOOST_DEFAULTS["use_best_model"]

    def test_make_classifier_catboost_noeval(self):
        """Test creating a CatBoost classifier without early stopping."""
        model = make_classifier("catboost_noeval", random_state=42)
        assert isinstance(model, CatBoostClassifier)
        params = model.get_params()
        assert params["iterations"] == CATBOOST_NOEVAL_DEFAULTS["iterations"]
        # catboost_noeval should not have use_best_model set
        assert "use_best_model" not in params or params.get("use_best_model") is None

    def test_make_classifier_rfe_rf(self):
        """Test creating RFE with Random Forest."""
        model = make_classifier("rfe_rf", random_state=42)
        assert isinstance(model, RFE)
        assert isinstance(model.estimator, RandomForestClassifier)
        assert model.n_features_to_select == 100
        assert model.step == 0.1

    def test_make_classifier_rfe_rf_custom_features(self):
        """Test RFE with custom n_features_to_select."""
        model = make_classifier("rfe_rf", random_state=42, n_features_to_select=50)
        assert isinstance(model, RFE)
        assert model.n_features_to_select == 50

    def test_make_classifier_rfe_catboost(self):
        """Test creating RFE with CatBoost."""
        model = make_classifier("rfe_catboost", random_state=42)
        assert isinstance(model, RFE)
        assert isinstance(model.estimator, CatBoostClassifier)

    def test_make_classifier_rfe_cv(self):
        """Test creating RFECV."""
        model = make_classifier("rfe_cv", random_state=42)
        assert isinstance(model, RFECV)
        assert isinstance(model.estimator, RandomForestClassifier)

    def test_make_classifier_invalid_type(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            make_classifier("invalid_type")

    def test_make_classifier_default_random_state(self):
        """Test that default random state is 42."""
        model = make_classifier("rf")
        assert model.random_state == 42
