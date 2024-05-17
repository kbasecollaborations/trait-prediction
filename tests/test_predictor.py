import pytest
from catboost import CatBoostClassifier

from trait_prediction.training import Predictor


def test_predictor_init(leaf_dataset_data, random_state):
    phenotype, feature = leaf_dataset_data
    classifier = CatBoostClassifier(
        random_state=random_state,
        objective="Logloss",
        verbose=False,
        allow_writing_files=False,
        task_type="CPU",
        thread_count=1,
    )
    predictor = Predictor(phenotype, feature, classifier, random_state=random_state)
    assert predictor.training_data is None
    assert predictor.cv_data is None


@pytest.mark.parametrize("sampling_type", ["random"])  # TODO: Add ooc
@pytest.mark.parametrize(
    "imbalance_correction", [None, "auto", "undersample", "oversample"]
)
def test_predictor_split_data(sampling_type, imbalance_correction, leaf_predictor):
    predictor = leaf_predictor
    predictor.split_data(
        sampling_type=sampling_type, imbalance_correction=imbalance_correction
    )
    assert predictor.training_data is not None
    assert predictor.cv_data is None
    assert (
        predictor.training_data.X_train.shape[0]
        == predictor.training_data.y_train.shape[0]
    )
    assert (
        predictor.training_data.X_test.shape[0]
        == predictor.training_data.y_test.shape[0]
    )
    n_samples = (
        predictor.training_data.X_train.shape[0]
        + predictor.training_data.X_test.shape[0]
    )
    if imbalance_correction not in ["undersample", "oversample"]:
        assert predictor.phenotype.phenotype_data.shape[0] == n_samples
    assert predictor.training_data.sampling_type == sampling_type


@pytest.mark.parametrize("stratify", [True, False])
def test_predictor_split_data_cv(stratify, leaf_predictor):
    predictor = leaf_predictor
    predictor.split_data_cv(n_splits=5, stratify=stratify)
    assert predictor.training_data is None
    assert predictor.cv_data is not None
    assert len(predictor.cv_data.folds) == 5


def test_predictor_fit_predict(leaf_predictor):
    predictor = leaf_predictor
    with pytest.raises(ValueError):
        predictor.fit()
    with pytest.raises(ValueError):
        predictor.predict()
    predictor.split_data(sampling_type="random", imbalance_correction=None)
    predictor.fit()
    y_pred = predictor.predict()
    assert y_pred.shape == predictor.training_data.y_test.shape


def test_predictor_get_score(leaf_predictor):
    predictor = leaf_predictor
    predictor.split_data(sampling_type="random", imbalance_correction=None)
    score = predictor.get_score(kind="test", n_jobs=1)
    assert len(score.estimators) == 1
    assert score.scores.shape[0] == 1


def test_predictor_get_score_cv(leaf_predictor):
    predictor = leaf_predictor
    predictor.split_data_cv(n_splits=5, stratify=True)
    score = predictor.get_score(kind="CV", n_jobs=1)
    assert len(score.estimators) == 5
    assert score.scores.shape[0] == 5
