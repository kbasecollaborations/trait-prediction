"""Classifiers module for trait prediction.

This module provides:
- A factory function for creating well-tuned classifiers
- Baseline classifiers for comparison (null models)
"""

from .bernoulli import BernoulliClassifier
from .classifier import Classifier
from .factory import (
    CATBOOST_DEFAULTS,
    CATBOOST_NOEVAL_DEFAULTS,
    DT_DEFAULTS,
    RF_DEFAULTS,
    RFE_DEFAULTS,
    RFECV_DEFAULTS,
    ModelType,
    make_classifier,
)
from .identity import IdentityClassifier
from .nearest_neighbor import NearestNeighborClassifier

__all__ = [
    # Factory function and types
    "make_classifier",
    "ModelType",
    # Default configurations
    "CATBOOST_DEFAULTS",
    "CATBOOST_NOEVAL_DEFAULTS",
    "RF_DEFAULTS",
    "DT_DEFAULTS",
    "RFE_DEFAULTS",
    "RFECV_DEFAULTS",
    # Base class
    "Classifier",
    # Baseline classifiers (null models)
    "BernoulliClassifier",
    "IdentityClassifier",
    "NearestNeighborClassifier",
]
