"""
feature_engineering.py
-----------------------
Functions for constructing the scikit-learn preprocessing pipeline that
performs encoding and scaling.

Responsibilities
----------------
- drop_id_columns              : Remove non-predictive identifier columns
- build_preprocessing_pipeline : Construct a ColumnTransformer pipeline that
                                 one-hot-encodes categoricals and standard-scales
                                 numericals

Design decisions
----------------
- The pipeline object is returned unfitted. Fitting happens in main.py on
  X_train only (fit_transform), and the same fitted pipeline is applied to
  X_test and new inference data (transform).
  
  This enforces the critical ML invariant: test data statistics never leak
  into the fitted transformers.

- The pipeline is a scikit-learn Pipeline, meaning it can be serialized with
  joblib and reloaded for inference without any code changes.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.config import CATEGORICAL_COLS, NUMERICAL_COLS, ID_COLUMNS


# ---------------------------------------------------------------------------
# 1. Drop non-predictive columns
# ---------------------------------------------------------------------------

def drop_id_columns(
    df: pd.DataFrame,
    id_columns: list[str] = ID_COLUMNS,
) -> pd.DataFrame:
    """
    Remove identifier columns that carry no predictive signal.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame (X), before or after splitting.
    id_columns : list[str]
        List of column names to drop (default: from config).

    Returns
    -------
    pd.DataFrame
        DataFrame without the identifier columns.
    """
    cols_to_drop = [col for col in id_columns if col in df.columns]
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# 2. Build preprocessing pipeline
# ---------------------------------------------------------------------------

def build_preprocessing_pipeline(
    categorical_cols: list[str] = CATEGORICAL_COLS,
    numerical_cols: list[str] = NUMERICAL_COLS,
) -> ColumnTransformer:
    """
    Construct a scikit-learn ColumnTransformer that preprocesses features.

    Numerical pipeline
    ------------------
    1. SimpleImputer (median strategy) — safety net for any remaining NaNs.
    2. StandardScaler — zero mean, unit variance.

    Categorical pipeline
    --------------------
    1. SimpleImputer (constant 'Unknown') — safety net for any NaN categories.
    2. OneHotEncoder — sparse output disabled, unknown categories ignored at
       inference time so unseen values don't crash the pipeline.

    The remainder of columns not listed in either list is dropped
    (e.g., datetime columns that have served their purpose).

    Parameters
    ----------
    categorical_cols : list[str]
        Column names to one-hot-encode (default: from config).
    numerical_cols : list[str]
        Column names to scale (default: from config).

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing pipeline ready to be fit_transformed on X_train
        and transformed on X_test / new data.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline,  numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",         # drop datetime columns and any other leftovers
        verbose_feature_names_out=False,
    )

    return preprocessor
