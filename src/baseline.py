"""
baseline.py
-----------
Baseline model implementation for the MEDILENS readmission prediction pipeline.

Purpose
-------
Establish a minimum performance benchmark using simple heuristics to demonstrate
that the primary ML model provides meaningful improvement over trivial solutions.

Baseline Strategy
-----------------
For binary classification (readmitted: 0/1), we use sklearn's DummyClassifier
with the 'most_frequent' strategy, which always predicts the majority class.

This baseline:
- Requires no feature engineering or model training
- Represents the simplest possible prediction strategy
- Provides a lower bound for model performance
- Helps identify if the dataset has predictive signal

Why This Baseline?
------------------
In healthcare readmission prediction:
- Readmission rates are typically imbalanced (fewer readmissions than non-readmissions)
- A majority-class baseline reveals whether the model learns meaningful patterns
- If the model cannot beat this baseline, it has no practical value
- The baseline anchors evaluation and prevents misleading accuracy claims

Design Notes
------------
- Baseline is fit ONLY on training data (no leakage)
- Uses identical evaluation metrics as the main model
- No feature preprocessing required (DummyClassifier ignores features)
- Reproducible with random_state parameter
"""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from src.config import RANDOM_STATE


def train_baseline_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    strategy: str = "most_frequent",
    random_state: int = RANDOM_STATE,
) -> DummyClassifier:
    """
    Fit a baseline DummyClassifier on training data.

    The DummyClassifier provides a simple baseline that makes predictions
    without learning from features. This establishes minimum performance
    that any useful model must exceed.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed feature matrix. Note: DummyClassifier ignores features,
        but we accept X_train for API consistency with train_model().
    y_train : pd.Series
        Binary target labels (0 = not readmitted, 1 = readmitted).
    strategy : str
        Prediction strategy for DummyClassifier:
        - 'most_frequent': Always predict majority class (default)
        - 'stratified': Random predictions matching training class distribution
        - 'uniform': Random predictions with equal probability
        - 'constant': Always predict a specified constant value
    random_state : int
        Random seed for reproducibility (used by 'stratified' and 'uniform').

    Returns
    -------
    DummyClassifier
        Fitted baseline model ready for evaluation.

    Raises
    ------
    ValueError
        If X_train and y_train have different numbers of rows.

    Examples
    --------
    >>> baseline = train_baseline_model(X_train, y_train)
    >>> y_pred = baseline.predict(X_test)
    >>> accuracy = (y_pred == y_test).mean()
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    baseline = DummyClassifier(strategy=strategy, random_state=random_state)
    baseline.fit(X_train, y_train)

    return baseline


def get_baseline_description(
    y_train: pd.Series,
    strategy: str = "most_frequent",
) -> dict:
    """
    Generate a description of the baseline strategy and training data distribution.

    This helps document what the baseline represents and why it was chosen.

    Parameters
    ----------
    y_train : pd.Series
        Training target labels.
    strategy : str
        Baseline strategy used.

    Returns
    -------
    dict
        Dictionary containing:
        - strategy: Baseline strategy name
        - description: Human-readable explanation
        - class_distribution: Training class counts
        - majority_class: Most frequent class in training data
        - majority_percentage: Percentage of majority class
        - is_imbalanced: Whether classes are imbalanced (>60% majority)

    Examples
    --------
    >>> desc = get_baseline_description(y_train)
    >>> print(desc['description'])
    'Always predicts class 0 (majority class in training data)'
    """
    # Calculate class distribution
    class_counts = y_train.value_counts().to_dict()
    total = len(y_train)
    majority_class = y_train.mode()[0]
    majority_count = class_counts[majority_class]
    majority_pct = (majority_count / total) * 100

    # Determine if imbalanced (using 60/40 threshold)
    is_imbalanced = majority_pct > 60.0

    # Generate description based on strategy
    descriptions = {
        "most_frequent": f"Always predicts class {majority_class} (majority class in training data)",
        "stratified": f"Random predictions matching training distribution: {class_counts}",
        "uniform": "Random predictions with equal probability for each class",
    }

    return {
        "strategy": strategy,
        "description": descriptions.get(strategy, f"Uses '{strategy}' strategy"),
        "class_distribution": class_counts,
        "majority_class": int(majority_class),
        "majority_percentage": round(majority_pct, 2),
        "is_imbalanced": is_imbalanced,
        "total_samples": total,
    }
