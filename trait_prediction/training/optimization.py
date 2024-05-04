import optuna


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
