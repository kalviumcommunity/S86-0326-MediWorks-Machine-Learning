"""
evaluate.py
-----------
Model evaluation function for the MEDILENS readmission prediction pipeline.

Responsibility
--------------
Given a fitted model and held-out test data, compute classification metrics
and return them as a dictionary.

Design notes
------------
- Returns a dict, never prints. Callers (main.py, notebooks) decide how to
  display or persist the metrics.
- Accepts the raw fitted model — does NOT import or call train.py. Training
  and evaluation are independent concerns.
- Includes ROC-AUC, which requires predict_proba; the function assumes the
  model supports probability estimates.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted classification model on held-out test data.

    Computes the following metrics:
        - accuracy   : fraction of correct predictions
        - precision  : positive predictive value (readmission correctly flagged)
        - recall     : sensitivity (fraction of actual readmissions detected)
        - f1         : harmonic mean of precision and recall
        - roc_auc    : area under the ROC curve (requires predict_proba support)
        - confusion_matrix : [[TN, FP], [FN, TP]]

    Parameters
    ----------
    model : sklearn estimator
        Any fitted classifier that implements predict() and predict_proba().
    X_test : np.ndarray
        Preprocessed feature matrix — must NOT have been seen during fitting.
    y_test : pd.Series
        True binary labels aligned with X_test rows.

    Returns
    -------
    dict
        Metrics dictionary with keys:
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
        'confusion_matrix', 'classification_report'.

    Raises
    ------
    ValueError
        If X_test and y_test have different numbers of rows.
    """
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Shape mismatch: X_test has {len(X_test)} rows but "
            f"y_test has {len(y_test)} rows."
        )

    predictions   = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]   # probability of class 1

    metrics = {
        "accuracy":               round(float(accuracy_score(y_test, predictions)),  4),
        "precision":              round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall":                 round(float(recall_score(y_test, predictions, zero_division=0)),    4),
        "f1":                     round(float(f1_score(y_test, predictions, zero_division=0)),        4),
        "roc_auc":                round(float(roc_auc_score(y_test, probabilities)),  4),
        "confusion_matrix":       confusion_matrix(y_test, predictions).tolist(),
        "classification_report":  classification_report(y_test, predictions, output_dict=True),
    }

    return metrics
