import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import catboost as cb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
from io import StringIO

import shap

from IPython.display import display
from IPython.core.display import HTML


classifier = "catboost"
phenotype = "gram_stain"


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1176, stratify=y
)

# Compute the class ratio
minority_class_count = sum(y_train == 1)  # Assuming '1' denotes the minority class
majority_class_count = len(y_train) - minority_class_count
class_ratio = minority_class_count / majority_class_count

# If the class ratio is too low (e.g., below 0.75), perform over-sampling
if class_ratio < 0.75:
    ros = RandomOverSampler(sampling_strategy=0.75, random_state=1176)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)  # type: ignore
else:
    # If not, simply keep the original samples
    X_train_resampled, y_train_resampled = X_train, y_train

# If the class ratio after over-sampling exceeds a threshold (e.g., greater than 0.9), perform under-sampling
if class_ratio > 0.9:
    rus = RandomUnderSampler(sampling_strategy=1.0, random_state=1176)
    X_train_resampled, y_train_resampled = rus.fit_resample(  # type: ignore
        X_train_resampled, y_train_resampled
    )


model = cb.CatBoostClassifier()
params = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "verbose": False,
}

# Train the classifier
model.set_params(**params)
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)


result = (
    "\t".join(
        [
            "Classifier",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "Balanced Accuracy",
            "Confusion Matrix\n",
        ]
    )
    + "\n"
)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

confusion_matrix_info = confusion_matrix(y_test, y_pred)

# Extracting values
TP = confusion_matrix_info[0, 0]
FP = confusion_matrix_info[0, 1]
FN = confusion_matrix_info[1, 0]
TN = confusion_matrix_info[1, 1]

confusion_matrix_data = (
    "TP=" + str(TP) + " TN=" + str(TN) + " FP=" + str(FP) + "FN=" + str(FN)
)


result += (
    "\t".join(
        [
            str(classifier),
            str(accuracy),
            str(precision),
            str(recall),
            str(f1),
            str(balanced_accuracy),
            str(confusion_matrix_data),
        ]
    )
    + "\n"
)


sio = StringIO(result)
result_df = pd.read_csv(sio, sep="\t")


# SHAP summary plot


# Create explainer object
explainer = shap.Explainer(model)

# Calculate SHAP values for all instances in your dataset
shap_values = explainer.shap_values(X_train_resampled)  # type: ignore

plt.switch_backend("Agg")
plt.figure(figsize=(20, 10))  # Adjust width (20) and height (10) as desired


title = classifier + "__" + phenotype
img_path = title + ".png"
plt.title(title)
shap.summary_plot(shap_values, X_train_resampled, max_display=20, plot_size=None)
plt.savefig(img_path)
plt.close()


# Classification report
print(classification_report(y_test, y_pred))
# Display results in table format
display(result_df)
# Shap summar plot
display(HTML("<img src='" + img_path + "'/>"))
