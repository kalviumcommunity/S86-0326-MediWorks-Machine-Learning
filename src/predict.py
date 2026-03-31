"""
predict.py
----------
Inference function for the MEDILENS readmission prediction pipeline.

Responsibility
--------------
Load saved artifacts (model + preprocessing pipeline) and generate predictions
on new, unseen patient visit data.

Design notes
------------
- predict() calls pipeline.transform(), NOT pipeline.fit_transform().
  This is the most critical invariant in the inference pipeline:
    * fit_transform() would refit the scaler/encoder on new data, producing
      different statistics from training time and breaking the model.
    * transform() applies the already-fitted transformers, exactly replicating
      the transformations seen at training time.
- The function never imports train.py — prediction is independent of training.
- Input validation ensures new data contains the columns the pipeline expects,
  giving a clear error instead of a cryptic sklearn shape mismatch.
"""

import pandas as pd

from src.data_preprocessing import validate_schema
from src.persistence import load_artifacts
from src.feature_engineering import drop_id_columns
from src.config import MODEL_PATH, PIPELINE_PATH, CATEGORICAL_COLS, NUMERICAL_COLS, ID_COLUMNS


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    new_data: pd.DataFrame,
    model_path: str = MODEL_PATH,
    pipeline_path: str = PIPELINE_PATH,
) -> pd.DataFrame:
    """
    Generate readmission risk predictions on new patient visit data.

    The function:
    1. Loads the fitted model and preprocessing pipeline from disk.
    2. Drops non-predictive ID columns.
    3. Applies pipeline.transform() (NOT fit_transform) to the new data.
    4. Generates binary predictions and readmission probability scores.
    5. Returns a DataFrame with predictions attached to the original data.

    Parameters
    ----------
    new_data : pd.DataFrame
        Raw patient visit DataFrame with the same schema as training data
        (minus the target column, which is unknown at inference time).
    model_path : str
        Path to the serialized model artifact (default: from config).
    pipeline_path : str
        Path to the serialized preprocessing pipeline (default: from config).

    Returns
    -------
    pd.DataFrame
        A copy of `new_data` with two additional columns:
        - 'predicted_readmission'      : 0 or 1
        - 'readmission_probability'    : float in [0.0, 1.0]

    Raises
    ------
    FileNotFoundError
        If model or pipeline artifacts are not found on disk.
    ValueError
        If required feature columns are missing from `new_data`.
    """
    # 1. Load artifacts (model and pipeline)
    model, pipeline = load_artifacts(
        model_path=model_path, pipeline_path=pipeline_path)

    # 2. Validate required feature columns are present
    validate_schema(new_data, required_columns=(
        NUMERICAL_COLS + CATEGORICAL_COLS))

    # 3. Drop ID columns (not part of the model's feature space)
    X_new = drop_id_columns(new_data.copy(), id_columns=ID_COLUMNS)

    # 4. Apply pipeline — TRANSFORM only, never fit
    #    This ensures the same scaling and encoding as at training time.
    X_new_processed = pipeline.transform(X_new)

    # 5. Generate predictions and probabilities
    predictions = model.predict(X_new_processed)
    probabilities = model.predict_proba(X_new_processed)[:, 1]

    # 6. Attach results to a copy of the original input for traceability
    result = new_data.copy()
    result["predicted_readmission"] = predictions
    result["readmission_probability"] = probabilities.round(4)

    return result
