"""
train.py
--------
Model training function for the MEDILENS readmission prediction pipeline.

Responsibility
--------------
Fit a RandomForestClassifier on pre-processed training data and return the
fitted model object.

Design notes
------------
- Training is entirely separate from evaluation (evaluate.py) and persistence
  (persistence.py). This function returns an artifact; callers decide what to
  do with it.
- `random_state` is an explicit parameter so every call is reproducible and
  testable without relying on global state.
- No printing inside the function — callers handle logging/reporting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.config import MODEL_PARAMS, RANDOM_STATE


# ---------------------------------------------------------------------------
# Model Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: pd.Series,
    model_params: dict = MODEL_PARAMS,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """
    Fit a RandomForestClassifier on the provided training data.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed feature matrix (output of pipeline.fit_transform).
    y_train : pd.Series
        Binary target labels aligned with X_train rows.
    model_params : dict
        Hyperparameter dict passed directly to RandomForestClassifier
        (default: from config.MODEL_PARAMS). Any key recognised by
        RandomForestClassifier is accepted, making this future-proof for
        hyperparameter tuning experiments.
    random_state : int
        Random seed for reproducibility (default: 42). Overrides the
        random_state key inside `model_params` if both are provided, ensuring
        the explicit argument takes precedence.

    Returns
    -------
    RandomForestClassifier
        Fitted model ready for evaluation or persistence.

    Raises
    ------
    ValueError
        If X_train and y_train have different numbers of rows.
    """
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Shape mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    # Merge params, with random_state argument taking precedence
    params = {**model_params, "random_state": random_state}

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    return model
