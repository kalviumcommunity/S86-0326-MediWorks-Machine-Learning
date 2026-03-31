"""
data_preprocessing.py
----------------------
Functions for loading raw hospital visit data, cleaning it, and splitting it
into training and test sets.

Responsibilities
----------------
- load_data        : Read CSV from disk and parse datetimes
- validate_schema  : Assert required columns are present
- clean_data       : Handle missing values and derive engineered base columns
- split_data       : Stratified train/test split with reproducible random state

Each function follows the single-responsibility principle:
one logical operation, explicit inputs, explicit outputs, no side effects.
"""

import os
from collections.abc import Sequence
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    REQUIRED_COLUMNS,
    DATETIME_COLS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_data(filepath: str, datetime_cols: list[str] = DATETIME_COLS) -> pd.DataFrame:
    """
    Load raw hospital visit data from a CSV file.

    Parses specified columns as datetime so downstream functions can compute
    derived time-based features (e.g., length of stay).

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the raw CSV file.
    datetime_cols : list[str]
        Column names to parse as datetime (default: from config).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with datetime columns properly parsed.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at `filepath`.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Place hospital_visits.csv in data/raw/ before running the pipeline."
        )

    df = pd.read_csv(filepath, parse_dates=datetime_cols)
    return df


# ---------------------------------------------------------------------------
# 2. Schema Validation
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, required_columns: Sequence[str] = REQUIRED_COLUMNS) -> None:
    """
    Validate that the DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list[str]
        Column names that must be present (default: from config).

    Raises
    ------
    ValueError
        With a descriptive message listing exactly which columns are missing.
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )


# ---------------------------------------------------------------------------
# 3. Data Cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """
    Clean the raw DataFrame by applying safe, non-leaky transformations.

     Operations performed (in order):
     1. Drop duplicate rows.
     2. Drop rows with missing target labels.
     3. Derive `length_of_stay` (days) from admission/discharge datetimes if not
         already present.

     Important
     ---------
     This function intentionally does NOT impute feature missing values.
     Imputation belongs inside the scikit-learn preprocessing pipeline so that
     statistics (like medians) are learned on training data only, preventing
     train/test leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (output of load_data).
    target_column : str
        Name of the target column. Rows with missing target values are dropped.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering and splitting.
    """
    df = df.copy()

    # 1. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Drop rows without labels
    if target_column in df.columns:
        df = df.dropna(subset=[target_column])

    # 2. Derive length_of_stay if not already a column
    if "length_of_stay" not in df.columns:
        if "admission_date" in df.columns and "discharge_date" in df.columns:
            df["length_of_stay"] = (
                (df["discharge_date"] - df["admission_date"]).dt.total_seconds()
                / 86_400  # convert seconds → days
            ).round(2)

    return df


# ---------------------------------------------------------------------------
# 4. Train / Test Split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the cleaned DataFrame into stratified training and test sets.

    Stratification on the target column ensures that both splits maintain the
    same class distribution — critical for imbalanced medical datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (output of clean_data).
    target_column : str
        Name of the binary target column (default: from config).
    test_size : float
        Proportion of the dataset used for testing (default: 0.20).
    random_state : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    X_train : pd.DataFrame
    X_test  : pd.DataFrame
    y_train : pd.Series
    y_test  : pd.Series
        Four-element tuple in the order documented above.

    Raises
    ------
    KeyError
        If `target_column` is not present in `df`.
    """
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
