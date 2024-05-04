"""Module that defines the Predictor class."""

import pathlib
import pickle
from typing import Iterable, NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from ..main import Feature, FeatureIndex, Phenotype, PhenotypeIndex
from .sampling import (
    perform_imbalanced_sampling,
    perform_ooc_sampling,
    perform_random_sampling,
)


class TrainingData(NamedTuple):
    """The training and testing data for the machine learning.

    Attributes
    ----------
    X_train : pd.DataFrame
        The training feature data.
    X_test : pd.DataFrame
        The testing feature data.
    y_train : pd.Series
        The training target data.
    y_test : pd.Series
        The testing target data.
    test_size : float
        The size of the test set.
    stratified : bool
        Whether the data was stratified.
    sampling_type : str
        The type of sampling used.
    imbalance_sampling_type : str | None
        The type of imbalanced sampling used.
    """

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    test_size: float
    stratified: bool
    sampling_type: str
    imbalanced_sampling_type: str | None


class CVData(NamedTuple):
    """The training and testing data for cross validation."""

    CVClass: KFold | StratifiedKFold
    folds: Iterable[tuple[np.ndarray, np.ndarray]]
    stratified: bool


class Score(NamedTuple):
    """The scores obtained from get_scores method"""

    pindex: PhenotypeIndex
    findex: FeatureIndex
    kind: str
    scores: pd.DataFrame
    estimators: list


class Predictor:
    """Class that performs machine learning on the phenotype and feature data.

    Parameters
    ----------
    classifier : Any
        The classifier used for the machine learning.
    phenotype : Phenotype
        The phenotype data.
    feature : Feature
        The feature data.
    random_state : int | None
        The random state for the machine learning.

    Attributes
    ----------
    phenotype : Phenotype
        Phenotype class containing the data for one phenotype.
    feature : Feature
        Feature class containing the data for one feature.
    classifier : Any
        Classifier used for the machine learning.
    random_state : int | None
        Random state for the machine learning.
    training_data : TrainingData | None
        Training data for the machine learning.
    """

    def __init__(
        self,
        phenotype: Phenotype,
        feature: Feature,
        classifier,
        random_state: int | None,
    ) -> None:
        self.phenotype = phenotype
        self.feature = feature
        self._check_inputs()
        self.classifier = classifier
        self.random_state = random_state
        self.training_data: TrainingData | None = None
        self.cv_data: CVData | None = None

    def _check_inputs(self) -> None:
        """Check if the indices of the phenotype and feature data match."""
        pd_index = self.phenotype.phenotype_data.index
        fd_index = self.feature.feature_data.index
        if not (pd_index == fd_index).all():
            raise ValueError("Phenotype and feature data indices do not match.")

    def split_data(
        self,
        sampling_type: str,
        test_size: float = 0.3,
        stratify: bool = True,
        imbalanced: str | None = None,
    ) -> None:
        """
        Split the data into train and test sets using `train_test_split`.

        Parameters
        ---------
        sampling_type : str
            Type of sampling to use.
        test_size : float
            Size of the test set.
            Default value is 0.3.
        stratify : bool
            Whether to stratify the data.
            Default value is True.
        imbalanced : Optional[str], {'auto', 'undersample', 'oversample'}
            Whether to use imbalanced data.
            Default value is 'auto'.

        Returns
        ------
        dict[str, pd.DataFrame | pd.Series]
            Dictionary containing the data for the machine learning.
        """
        X = self.feature.feature_data
        y = self.phenotype.phenotype_data
        random_state = self.random_state
        if np.isclose(test_size, 0.0):
            X_train = X
            X_test = pd.DataFrame()
            y_train = y
            y_test = pd.Series()
        else:
            if sampling_type == "random":
                X_train, X_test, y_train, y_test = perform_random_sampling(
                    X, y, test_size, stratify, random_state
                )
            elif sampling_type == "ooc":
                X_train, X_test, y_train, y_test = perform_ooc_sampling(
                    X, y, test_size, stratify, random_state
                )
            else:
                raise ValueError("sampling_type must be 'random' or 'ooc'.")
        if imbalanced is not None:
            imbalanced_sampling_type, X_train, y_train = perform_imbalanced_sampling(
                y, X_train, y_train, imbalanced, random_state
            )
        else:
            imbalanced_sampling_type = None
        self.training_data = TrainingData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            test_size=test_size,
            stratified=stratify,
            sampling_type=sampling_type,
            imbalanced_sampling_type=imbalanced_sampling_type,
        )

    def split_data_cv(self, n_splits: int, stratify: bool) -> None:
        """Split the data into train and test sets using cross validation.

        Parameters
        ----------
        n_splits : int
            The number of splits.
        stratify : bool
            Whether to stratify the data.
        """
        X = self.feature.feature_data
        y = self.phenotype.phenotype_data
        if stratify:
            CVClass = StratifiedKFold
        else:
            CVClass = KFold
        cv = CVClass(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        self.cv_data = CVData(
            CVClass=cv, folds=list(cv.split(X, y)), stratified=stratify
        )

    def fit(self) -> None:
        """Fit the classifier to the training data."""
        if self.training_data is not None:
            X_train = self.training_data.X_train
            y_train = self.training_data.y_train
            self.classifier.fit(X_train, y_train)
        else:
            raise ValueError("Data has not been prepared. Call `split_data` first.")

    def predict(self, X: pd.DataFrame | None = None) -> pd.Series:
        """
        Predict the phenotype of the given feature data.

        Parameters
        ---------
        X : Optional[pd.DataFrame]
            Pandas DataFrame containing the feature data.
            Default value is None.
            If None, then self.training_data is used.

        Returns
        ------
        pd.Series
            Pandas DataFrame containing the predicted phenotype.
        """
        if X is not None:
            y_pred = self.classifier.predict(X)
        else:
            if self.training_data is None:
                raise ValueError("Data has not been prepared. Call `split_data` first.")
            X_test = self.training_data.X_test
            y_pred = self.classifier.predict(X_test)
        return y_pred

    def get_score(
        self,
        kind: str = "CV",
        n_jobs: int = -1,
        scoring: Iterable[str] = (
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "matthews_corrcoef",
        ),
    ) -> Score:
        """Get the scores of the classifier from CV or test data.
        Note: This method trains the classifier on the training data or CV data, but original classifier is not modified.

        Parameters
        ----------
        kind : str
            The type of data to use. Either 'CV' or 'test'.
        n_jobs : int
            The number of jobs to run in parallel.
        scoring : Iterable[str]
            The scoring metrics to use.

        Returns
        -------
        Score
            The scores obtained from the classifier along with the estimators.
        """
        X = self.feature.feature_data
        y = self.phenotype.phenotype_data
        clf = self.classifier
        if kind == "CV":
            if self.cv_data is None:
                raise ValueError(
                    "Data has not been prepared. Call `split_data_cv` first."
                )
            cv = self.cv_data.folds
        elif kind == "test":
            if self.training_data is None:
                raise ValueError("Data has not been prepared. Call `split_data` first.")
            train_indices = self.training_data.X_train.index
            train_iloc = [X.index.get_loc(i) for i in train_indices]
            test_indices = self.training_data.X_test.index
            test_iloc = [X.index.get_loc(i) for i in test_indices]
            cv = [(train_iloc, test_iloc)]
        else:
            raise ValueError("kind must be 'CV' or 'test'.")
        cv_results = cross_validate(
            clf,
            X,
            y,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            return_estimator=True,
            return_indices=False,
        )
        scores_data = {
            k.lstrip("test_"): v for k, v in cv_results.items() if k.startswith("test_")
        }
        for train_indices, test_indices in cv:
            y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
            train_class_counts = y_train.value_counts()
            test_class_counts = y_test.value_counts()
            scores_data["train_class_0"] = train_class_counts.get(0, 0)
            scores_data["train_class_1"] = train_class_counts.get(1, 0)
            scores_data["test_class_0"] = test_class_counts.get(0, 0)
            scores_data["test_class_1"] = test_class_counts.get(1, 0)
        scores_df = pd.DataFrame(scores_data)
        pindex = self.phenotype.pindex
        findex = self.feature.findex
        scores = Score(
            pindex=pindex,
            findex=findex,
            kind=kind,
            scores=scores_df,
            estimators=cv_results["estimator"],
        )
        return scores

    def save(self, folder: str | pathlib.Path) -> None:
        """Save the current state of the Predictor.

        Parameters
        ---------
        folder : str | pathlib.Path
            Folder to save the objects
        """
        folder = pathlib.Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        predictor_file = folder / "predictor.pkl"
        with open(predictor_file, "wb") as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "Predictor":
        """
        Load the Predictor object from pkl file

        Parameters
        ---------
        file_path : str | pathlib.Path
            Path to the pkl file

        Returns
        ------
        Predictor
            Predictor object
        """
        predictor_file = file_path
        with open(predictor_file, "rb") as fid:
            phenotype_predictor = pickle.load(fid)
        return phenotype_predictor
