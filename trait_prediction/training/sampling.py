from collections import Counter
from typing import Tuple

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def perform_imbalanced_sampling(
    y: pd.Series,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    imbalanced: str,
    random_state: int | None,
) -> Tuple[str | None, pd.DataFrame, pd.Series]:
    if imbalanced == "auto":
        counter = Counter(y)
        majority_class_count = counter.most_common()[0][1]
        minority_class_count = counter.most_common()[-1][1]
        class_ratio = minority_class_count / majority_class_count
        # NOTE: perform sampling if class_ratio is less than 0.2
        if class_ratio <= 0.2:
            # FIXME: This parameter should be optimized
            if minority_class_count <= 25:
                # then we have a small minority class so we do oversampling
                # TODO: Replace this with SMOTEN?
                sampler = RandomOverSampler(random_state=random_state)
                sampling_type = "oversample"
            else:
                sampler = RandomUnderSampler(random_state=random_state)
                sampling_type = "undersample"
        else:
            sampler = None
            sampling_type = None
    elif imbalanced == "undersample":
        # NOTE: this removes samples from the majority class
        sampler = RandomUnderSampler(random_state=random_state)
        sampling_type = "undersample"
    elif imbalanced == "oversample":
        # NOTE: this adds samples to the minority class (random sample with replacement)
        sampler = RandomOverSampler(random_state=random_state)
        sampling_type = "oversample"
    else:
        raise ValueError("imbalanced must be 'auto', 'undersample', or 'oversample'.")
    if sampler is not None:
        X_train_new, y_train_new = sampler.fit_resample(X_train, y_train)  # type: ignore
    else:
        X_train_new, y_train_new = X_train, y_train
    return sampling_type, X_train_new, y_train_new  # type: ignore


def perform_random_sampling(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    stratify: bool,
    random_state: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if stratify:
        y_stratify = y
    else:
        y_stratify = None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y_stratify,
    )
    return X_train, X_test, y_train, y_test  # type: ignore


# TODO: perform_ooc_sampling
def perform_ooc_sampling(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    stratify: bool,
    random_state: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    pass
