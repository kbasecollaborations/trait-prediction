"""Classifier factory and null-model baselines.

``make_classifier`` and the null models (``BernoulliClassifier``,
``IdentityClassifier``, ``NearestNeighborClassifier``) are used directly by the
manuscript analysis scripts.
"""

import pandas as pd
import pytest
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier

from trait_prediction.classifiers import (
    BernoulliClassifier,
    IdentityClassifier,
    NearestNeighborClassifier,
    make_classifier,
)


@pytest.mark.parametrize(
    "model_type,expected_cls",
    [
        ("rf", RandomForestClassifier),
        ("dt", DecisionTreeClassifier),
        ("catboost", CatBoostClassifier),
        ("rfe_rf", RFE),
    ],
)
def test_make_classifier_types(model_type, expected_cls):
    model = make_classifier(model_type, random_state=42)
    assert isinstance(model, expected_cls)


def test_make_classifier_custom_params():
    model = make_classifier("rf", random_state=42, n_estimators=37)
    assert model.n_estimators == 37


def test_make_classifier_invalid():
    with pytest.raises(ValueError, match="Unknown model type"):
        make_classifier("not_a_model")


@pytest.fixture
def xy():
    X = pd.DataFrame(
        {"a": [0, 1, 0, 1, 0, 1], "b": [1, 0, 1, 0, 1, 0]},
        index=[f"g{i}" for i in range(6)],
    )
    y = pd.Series([1, 1, 1, 0, 1, 1], index=X.index)  # class 1 is the majority
    return X, y


def test_identity_classifier_predicts_majority(xy):
    X, y = xy
    clf = IdentityClassifier(random_state=42, categorical_feature_names=[])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert (preds == 1).all()


def test_bernoulli_classifier_is_binary(xy):
    X, y = xy
    clf = BernoulliClassifier(random_state=42, categorical_feature_names=[])
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds.unique()).issubset({0, 1})


def test_nearest_neighbor_classifier(xy):
    X, y = xy
    newick = "((g0,g1),(g2,g3),(g4,g5));"
    clf = NearestNeighborClassifier(
        random_state=42, categorical_feature_names=[], tree=newick
    )
    clf.fit(X, y)
    preds = clf.predict(X)
    assert len(preds) == len(X)
    assert set(preds.dropna().unique()).issubset({0, 1})
