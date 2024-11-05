#!/usr/bin/env python

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict probabilities
probabilities = clf.predict_proba(X_test)


def bootstrap_confidence_interval(
    data, num_bootstrap_samples=1000, confidence_level=0.95
):
    bootstrap_samples = np.random.choice(
        data, (num_bootstrap_samples, len(data)), replace=True
    )
    bootstrap_means = np.mean(bootstrap_samples, axis=1)
    lower_bound = np.percentile(bootstrap_means, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_means, (1 + confidence_level) / 2 * 100)
    return lower_bound, upper_bound


# Example for the first test sample and first class
sample_probabilities = probabilities[:, 0]  # Probabilities for class 0
lower, upper = bootstrap_confidence_interval(sample_probabilities)
print(f"Confidence interval for class 0: [{lower}, {upper}]")
