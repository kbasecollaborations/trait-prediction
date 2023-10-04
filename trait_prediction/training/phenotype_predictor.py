"""Module that defines the PhenotypePredictor class."""

import pathlib
import pickle
from typing import Optional

import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_validate, train_test_split

from ..main import Phenotype

# TODO: hyperparam optimization using optuna
# TODO: Use logging module instead of print statements


class PhenotypePredictor:
    """
    Perform machine learning on Phenotype class and visualize the results.

    Parameters
    ---------
    phenotype : Phenotype
        Phenotype class containing the data for one phenotype.
    classifier : any
        Classifier used for the machine learning.
    random_state : Optional[int]
        Random state for the machine learning.

    Attributes
    ---------
    phenotype : Phenotype
        Phenotype class containing the data for one phenotype.
    classifier : any
        Classifier used for the machine learning.
    random_state : Optional[int]
        Random state for the machine learning.
    data : dict[str, pd.DataFrame | pd.Series]
        Dictionary containing the data for the machine learning.
    """

    def __init__(
        self, phenotype: Phenotype, classifier, random_state: Optional[int] = None
    ) -> None:
        self.phenotype = phenotype
        self.classifier = classifier
        self.random_state = random_state
        if phenotype.feature_data is None:
            raise ValueError(
                "Phenotype does not contain feature data. Attach feature data using `phenotype.set_feature_data`."
            )
        self._X = self.phenotype.feature_data
        self._y = self.phenotype.phenotype_data
        self._data_prep = False

    @property
    def data(self) -> dict[str, pd.DataFrame | pd.Series]:
        """Dictionary containing the data for the machine learning."""
        if self._data_prep:
            return {
                "X_train": self._X_train.copy(),
                "X_test": self._X_test.copy(),
                "y_train": self._y_train.copy(),
                "y_test": self._y_test.copy(),
            }
        else:
            raise ValueError("Data has not been prepared. Call `split_data` first.")

    def split_data(
        self,
        test_size: float = 0.3,
        stratify: bool = True,
        imbalanced: Optional[str] = "auto",
    ) -> dict[str, pd.DataFrame | pd.Series]:
        """
        Split the data into train and test sets using `train_test_split`.

        Parameters
        ---------
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
        if stratify:
            y_stratify = self._y
        else:
            y_stratify = None
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(
            self._X,
            self._y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y_stratify,
        )
        sampling_type = None
        if imbalanced is not None:
            if imbalanced == "auto":
                from collections import Counter

                counter = Counter(self._y)
                majority_class_count = counter.most_common()[0][1]
                minority_class_count = counter.most_common()[-1][1]
                class_ratio = minority_class_count / majority_class_count
                # perform sampling if class_ratio is less than 0.1
                if class_ratio <= 0.1:
                    # if minority_class_count has less than 100 data points
                    if minority_class_count < 100:
                        # then we have a small minority class so we do oversampling
                        self._sampler = RandomOverSampler(
                            random_state=self.random_state
                        )
                        sampling_type = "oversample"
                    else:
                        self._sampler = RandomUnderSampler(
                            random_state=self.random_state
                        )
                        sampling_type = "undersample"
                else:
                    self._sampler = RandomUnderSampler(random_state=self.random_state)
                    sampling_type = "undersample"
            elif imbalanced == "undersample":
                # NOTE: this removes samples from the majority class
                self._sampler = RandomUnderSampler(random_state=self.random_state)
                sampling_type = "undersample"
            elif imbalanced == "oversample":
                # NOTE: this adds samples to the minority class (random sample with replacement)
                self._sampler = RandomOverSampler(random_state=self.random_state)
                sampling_type = "oversample"
            else:
                raise ValueError(
                    "imbalanced must be 'auto', 'undersample', or 'oversample'."
                )
            # NOTE: Indices of the training dataset are lost after sampling
            self._X_train, self._y_train = self._sampler.fit_resample(  # type: ignore
                self._X_train, self._y_train
            )
        self._data_prep = True
        self._sampling_params = {
            "test_size": test_size,
            "stratify": stratify,
            "sampling_type": sampling_type,
        }
        return self.data

    def fit(self) -> None:
        """Fit the classifier to the training data."""
        if self._data_prep:
            self.classifier.fit(self._X_train, self._y_train)
        else:
            raise ValueError("Data has not been prepared. Call `split_data` first.")

    def predict(self, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Predict the phenotype of the given feature data.

        Parameters
        ---------
        X : Optional[pd.DataFrame]
            Pandas DataFrame containing the feature data.
            Default value is None.
            If None, then self._y_test is used.

        Returns
        ------
        pd.DataFrame
            Pandas DataFrame containing the predicted phenotype.
        """
        if X is not None:
            y_pred = self.classifier.predict(X)
        else:
            y_pred = self.classifier.predict(self._X_test)
        return y_pred

    def save(self, folder: str | pathlib.Path) -> None:
        """
        Save the current state of the PhenotypePredictor and Phenotype to a folder.

        Parameters
        ---------
        folder : str | pathlib.Path
            Folder to save the objects
        """
        folder = pathlib.Path(folder)
        folder.mkdir(exist_ok=True, parents=True)
        pt_file = folder / "phenotype.pkl"
        pt_predictor_file = folder / "phenotype_predictor.pkl"
        self.phenotype.save(pt_file)
        data = {
            "classifier": self.classifier,
            "random_state": self.random_state,
            "_data_prep": self._data_prep,
        }
        if self._data_prep is not None:
            data.update(
                **{
                    "classifier": self.classifier,
                    "random_state": self.random_state,
                    "sampling_params": self._sampling_params,
                    "data": self.data,
                }
            )
        with open(folder / "phenotype_predictor.pkl", "wb") as fid:
            pickle.dump(self, fid)

    @classmethod
    def load(cls, folder: str | pathlib.Path) -> "PhenotypePredictor":
        """
        Load the PhenotypePredictor and Phenotype from a folder.

        Parameters
        ---------
        folder : str | pathlib.Path
            Folder to load the objects

        Returns
        ------
        PhenotypePredictor
            PhenotypePredictor object
        """
        folder = pathlib.Path(folder)
        pt_file = folder / "phenotype.pkl"
        pt_predictor_file = folder / "phenotype_predictor.pkl"
        phenotype = Phenotype.load(pt_file)
        with open(pt_predictor_file, "rb") as fid:
            data = pickle.load(fid)
        phenotype_predictor = cls(phenotype, data["classifier"], data["random_state"])
        if data["_data_prep"] is not None:
            phenotype_predictor._data_prep = data["_data_prep"]
            phenotype_predictor._sampling_params = data["sampling_params"]
            phenotype_predictor._X_train = data["data"]["X_train"]
            phenotype_predictor._X_test = data["data"]["X_test"]
            phenotype_predictor._y_train = data["data"]["y_train"]
            phenotype_predictor._y_test = data["data"]["y_test"]
        return phenotype_predictor

    def cross_validate_kfold(
        self,
        n_splits: int,
        n_jobs: int = -1,
        scoring=(
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "matthews_corrcoef",
        ),
    ) -> tuple[pd.DataFrame, list]:
        """
        Perform cross validation using StratifiedKFold and return scores

        Parameters
        ---------
        n_splits : int
            Number of splits
        n_jobs : int, optional
            The number of jobs to run in parallel
            Default value is -1 (uses all available processors)
        scoring : tuple[str], optional
            The scoring metrics to use during cross validation
            Default value is ("balanced_accuracy", "precision", "recall", "f1", "roc_auc", "matthews_corrcoef").

        Returns
        ------
        cv_df: pd.DataFrame
            Cross validation scores
        estimators: list
            List of estimators for each fold
        """
        if self._data_prep:
            scores = cross_validate(
                self.classifier,
                self._X_train,  # type: ignore
                self._y_train,
                scoring=scoring,
                cv=n_splits,
                n_jobs=n_jobs,
                return_estimator=True,
            )
        else:
            raise ValueError("Data has not been prepared. Call `split_data` first.")
        scoring_metrics = [x for x in scores.keys() if x.startswith("test_")]
        data = []
        for metric in scoring_metrics:
            for i, score in enumerate(scores[metric]):
                data.append(
                    {
                        "metric": metric.strip("test_"),
                        "score": score,
                        "fold": i + 1,
                    }
                )
        cv_df = pd.DataFrame(data)
        estimators = scores["estimator"]
        return cv_df, estimators

    @staticmethod
    def plot_cross_validation(scores: pd.DataFrame):
        """
        Visualize cross validation performance

        Parameters
        ---------
        scores : pd.DataFrame
            Cross validation scores obtained from PhenotypePredictor.cross_validate_kfold
        """
        plot = sns.boxplot(scores, y="metric", x="score")
        plot.set_xlim((0, 1))
        return plot

    # TODO: Convert this to create_model method
    # Or use this: https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments
    def optimize(self, classifier, n_trials: int, direction: str = "maximize"):
        """
        Performs hyperparameter optimization using Optuna.

        Parameters
        ----------
        classifier : any
            Classifier used for the machine learning.
        n_trials : int
            Number of trials for the optimization.
        direction : {"maximize", "minimize"}
            Direction of optimization
            Default value is "maximize".
        """
        import shap

        def objective(trial: optuna.trial.Trial) -> float:
            k = trial.suggest_int("k", 100, self._X_train.shape[1], step=100)
            # NOTE: Only supports CatBoost params currently
            params = {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_uniform("learning_rate", 0.01, 0.5),
                "depth": trial.suggest_int("depth", 1, 10),
            }
            model_full = classifier(**params, random_state=self.random_state)
            model_full.fit(self._X_train, self._y_train)
            # perform feature selection using shap
            explainer = shap.Explainer(model_full)
            shap_values = explainer.shap_values(self._X_test)  # type: ignore
            shap_importances = np.abs(shap_values).mean(0)
            importance_df = pd.DataFrame(
                {"feature": self._X_train.columns, "importance": shap_importances}
            )
            importance_df.sort_values(by="importance", ascending=False, inplace=True)
            feature_list = importance_df["feature"].tolist()[:k]
            # update the model
            model = classifier(**params, random_state=self.random_state)
            model.fit(self._X_train[feature_list], self._y_train)
            cv_score = cross_validate(
                model,
                self._X_train[feature_list],  # type: ignore
                self._y_train,
                scoring=("matthews_corrcoef", "balanced_accuracy"),
                cv=5,
                n_jobs=-1,
            )
            return cv_score["test_matthews_corrcoef"].mean()

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        return study
