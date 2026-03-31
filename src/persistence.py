"""
persistence.py
--------------
Functions for saving and loading model artifacts (fitted model and
preprocessing pipeline) using joblib serialization.

Responsibilities
----------------
- save_artifacts : Serialize and persist model + pipeline to disk
- load_artifacts : Deserialize and return model + pipeline from disk

Design notes
------------
- Persistence is intentionally separated from training and prediction.
  train.py trains the model and returns it; main.py decides when and where
  to save it. predict.py loads artifacts and generates predictions; it does
  not retrain.
- joblib is preferred over pickle for scikit-learn objects: it handles
  large numpy arrays (like forest estimator trees) more efficiently.
"""

import os
import joblib

from src.config import MODEL_PATH, PIPELINE_PATH


# ---------------------------------------------------------------------------
# Save Artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    model,
    pipeline,
    model_path: str = MODEL_PATH,
    pipeline_path: str = PIPELINE_PATH,
) -> None:
    """
    Serialize and save the fitted model and preprocessing pipeline to disk.

    Creates parent directories if they do not already exist.

    Parameters
    ----------
    model : sklearn estimator
        Fitted classification model (e.g., RandomForestClassifier).
    pipeline : sklearn ColumnTransformer or Pipeline
        Fitted preprocessing pipeline.
    model_path : str
        File path for the serialized model (default: from config).
    pipeline_path : str
        File path for the serialized pipeline (default: from config).

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(model_path),    exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

    joblib.dump(model,    model_path)
    joblib.dump(pipeline, pipeline_path)


# ---------------------------------------------------------------------------
# Load Artifacts
# ---------------------------------------------------------------------------

def load_artifacts(
    model_path: str = MODEL_PATH,
    pipeline_path: str = PIPELINE_PATH,
) -> tuple:
    """
    Load previously saved model and preprocessing pipeline from disk.

    Parameters
    ----------
    model_path : str
        File path to the serialized model (default: from config).
    pipeline_path : str
        File path to the serialized pipeline (default: from config).

    Returns
    -------
    tuple
        (model, pipeline) — the fitted model and preprocessing pipeline,
        in that order.

    Raises
    ------
    FileNotFoundError
        If either artifact file does not exist at the specified path.
    """
    for path, label in [(model_path, "Model"), (pipeline_path, "Pipeline")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} artifact not found at '{path}'. "
                "Run main.py (training mode) first to generate and save artifacts."
            )

    model    = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)

    return model, pipeline
