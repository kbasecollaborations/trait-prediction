"""Classifier factory with well-tuned default hyperparameters.

This module provides a factory function for creating classifiers with sensible
defaults that have been tuned for genomic trait prediction tasks.
"""

from typing import Literal

import sklearn
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sklearn.set_config(enable_metadata_routing=True)

ModelType = Literal["rf", "dt", "catboost", "catboost_noeval", "rfe_rf", "rfe_catboost", "rfe_cv"]

# Well-tuned defaults for CatBoost with early stopping
CATBOOST_DEFAULTS: dict[str, int | float | str | bool] = {
    "iterations": 1000,
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 15,
    "bagging_temperature": 1,
    "rsm": 0.1,  # Only look at 10% of features per split for diversity
    "bootstrap_type": "Bayesian",
    "od_type": "Iter",
    "od_wait": 50,
    "task_type": "CPU",
    "eval_metric": "Logloss",
    "verbose": False,
    "allow_writing_files": False,
    "use_best_model": True,
    "thread_count": -1,
}

# CatBoost defaults without early stopping (for when no validation set)
CATBOOST_NOEVAL_DEFAULTS: dict[str, int | float | str | bool] = {
    "iterations": 500,  # Fixed iterations since no early stopping
    "learning_rate": 0.03,
    "depth": 4,
    "l2_leaf_reg": 15,  # Stronger regularization to prevent overfitting
    "bagging_temperature": 1,
    "rsm": 0.1,
    "bootstrap_type": "Bayesian",
    "task_type": "CPU",
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

# Random Forest defaults
RF_DEFAULTS: dict[str, int | float | str | None] = {
    "n_estimators": 1000,
    "max_depth": None,  # RF needs deep trees - let them grow
    "min_samples_split": 5,  # Slight regularization to prevent leaf purity on noise
    "min_samples_leaf": 2,  # Require at least 2 samples per leaf
    "max_features": "sqrt",  # Critical: distinct trees need feature subsets
    "n_jobs": -1,
}

# Decision Tree defaults
DT_DEFAULTS: dict[str, int | str | None] = {
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
}

# RFE defaults
RFE_DEFAULTS: dict[str, int | float] = {
    "step": 0.1,
    "n_features_to_select": 100,
    "verbose": 0,
}

# RFECV defaults
RFECV_DEFAULTS: dict[str, int | float | str] = {
    "step": 0.1,
    "min_features_to_select": 100,
    "scoring": "balanced_accuracy",
    "verbose": 0,
    "n_jobs": -1,
}


def make_classifier(
    model_type: ModelType = "catboost",
    random_state: int = 42,
    **kwargs: int | float | str | bool | None,
) -> BaseEstimator:
    """Create a classifier with well-tuned default hyperparameters.

    This factory function creates classifiers with defaults that have been
    optimized for genomic trait prediction tasks. All defaults can be
    overridden via keyword arguments.

    Parameters
    ----------
    model_type
        Type of classifier to create. Options:
        - "rf": Random Forest classifier
        - "dt": Decision Tree classifier
        - "catboost": CatBoost with early stopping (requires validation set)
        - "catboost_noeval": CatBoost without early stopping
        - "rfe_rf": Random Forest with Recursive Feature Elimination
        - "rfe_catboost": CatBoost with Recursive Feature Elimination
        - "rfe_cv": Cross-validated RFE with Random Forest
    random_state
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments to override default parameters.
        For RFE variants, use "n_features_to_select" to control feature count.

    Returns
    -------
    BaseEstimator
        Configured classifier instance ready for fitting.

    Raises
    ------
    ValueError
        If model_type is not one of the supported values.

    Examples
    --------
    >>> # Create CatBoost with early stopping
    >>> model = make_classifier("catboost", random_state=42)
    >>> model.fit(X_train, y_train, eval_set=(X_val, y_val))

    >>> # Create Random Forest with custom parameters
    >>> model = make_classifier("rf", n_estimators=500, max_depth=10)

    >>> # Create RFE with CatBoost selecting top 50 features
    >>> model = make_classifier("rfe_catboost", n_features_to_select=50)
    """
    if model_type == "rf":
        updated_kwargs = {**RF_DEFAULTS, "random_state": random_state, **kwargs}
        return RandomForestClassifier(**updated_kwargs)

    elif model_type == "dt":
        updated_kwargs = {**DT_DEFAULTS, "random_state": random_state, **kwargs}
        return DecisionTreeClassifier(**updated_kwargs)

    elif model_type == "catboost":
        updated_kwargs = {**CATBOOST_DEFAULTS, "random_state": random_state, **kwargs}
        return CatBoostClassifier(**updated_kwargs)

    elif model_type == "catboost_noeval":
        updated_kwargs = {**CATBOOST_NOEVAL_DEFAULTS, "random_state": random_state, **kwargs}
        return CatBoostClassifier(**updated_kwargs)

    elif model_type == "rfe_rf":
        # Extract RFE-specific kwargs
        rfe_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in RFE_DEFAULTS
        }
        updated_rfe_kwargs = {**RFE_DEFAULTS, **rfe_kwargs}
        estimator = make_classifier("rf", random_state=random_state, **kwargs)
        return RFE(estimator=estimator, **updated_rfe_kwargs)

    elif model_type == "rfe_catboost":
        # Extract RFE-specific kwargs
        rfe_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in RFE_DEFAULTS
        }
        updated_rfe_kwargs = {**RFE_DEFAULTS, **rfe_kwargs}
        estimator = make_classifier("catboost_noeval", random_state=random_state, **kwargs)
        return RFE(estimator=estimator, **updated_rfe_kwargs)

    elif model_type == "rfe_cv":
        # Extract RFECV-specific kwargs
        n_splits = kwargs.pop("n_splits", 5)
        rfecv_kwargs = {
            k: kwargs.pop(k) for k in list(kwargs.keys()) if k in RFECV_DEFAULTS
        }
        updated_rfecv_kwargs = {**RFECV_DEFAULTS, **rfecv_kwargs}
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        estimator = make_classifier("rf", random_state=random_state, **kwargs)
        return RFECV(estimator=estimator, cv=cv, **updated_rfecv_kwargs)

    else:
        valid_types = ", ".join(
            ["rf", "dt", "catboost", "catboost_noeval", "rfe_rf", "rfe_catboost", "rfe_cv"]
        )
        raise ValueError(f"Unknown model type: {model_type}. Must be one of: {valid_types}")
