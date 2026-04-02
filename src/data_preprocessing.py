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
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    REQUIRED_CORE_COLUMNS,
    DATETIME_COLS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERICAL_COLS,
    ALL_FEATURES,
    EXCLUDED_COLUMNS,
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

def validate_schema(
    df: pd.DataFrame,
    required_core_columns: list[str] = REQUIRED_CORE_COLUMNS,
    datetime_cols: list[str] = DATETIME_COLS,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Validate that the DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_core_columns : list[str]
        Columns that must always be present (default: from config).
    datetime_cols : list[str]
        Datetime columns used to derive length of stay (default: from config).
    target_column : str
        Name of the target column (default: from config).

    Raises
    ------
    ValueError
        With a descriptive message listing exactly which columns are missing.
    """
    missing_core = set(required_core_columns) - set(df.columns)
    if missing_core:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_core)}. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )

    # Ensure we can compute or use length_of_stay
    has_los = "length_of_stay" in df.columns
    has_datetimes = set(datetime_cols).issubset(set(df.columns))
    if not has_los and not has_datetimes:
        raise ValueError(
            "Dataset must contain either 'length_of_stay' OR both datetime columns "
            f"{datetime_cols} to derive it."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target '{target_column}' not found in dataset. "
            f"Found columns: {sorted(df.columns.tolist())}"
        )


# ---------------------------------------------------------------------------
# 3b. Feature / Target Separation (X, y)
# ---------------------------------------------------------------------------

def separate_features_and_target(
    df: pd.DataFrame,
    feature_columns: list[str] = ALL_FEATURES,
    target_column: str = TARGET_COLUMN,
    excluded_columns: list[str] = EXCLUDED_COLUMNS,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix (X) and target vector (y) with leak checks."""
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {df.columns.tolist()}"
        )

    if target_column in feature_columns:
        raise ValueError(
            "Target leaked into features: TARGET_COLUMN is in ALL_FEATURES")

    overlap = set(feature_columns).intersection(set(excluded_columns))
    if overlap:
        raise ValueError(
            f"Invalid configuration: features overlap with excluded columns: {sorted(overlap)}"
        )

    missing_features = set(feature_columns) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Dataset is missing required feature columns: {sorted(missing_features)}. "
            "Update ALL_FEATURES in config.py or fix the dataset schema."
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    if verbose:
        print(f"Features: {X.shape}")
        print(f"Target: {y.shape}")
        try:
            print(f"Target distribution:\n{y.value_counts(normalize=True)}")
        except Exception:
            pass

    return X, y


# ---------------------------------------------------------------------------
# 3. Data Cleaning
# ---------------------------------------------------------------------------

def clean_data(
    df: pd.DataFrame,
    numerical_cols: list[str] = NUMERICAL_COLS,
) -> pd.DataFrame:
    """
    Clean the raw DataFrame by handling missing values and deriving base columns.

    Operations performed (in order):
    1. Drop duplicate rows.
    2. Derive `length_of_stay` (days) from admission/discharge datetimes if not
       already present.
    3. Fill numerical column NaNs with the column median (computed on this
       DataFrame — no leakage risk since this is called before splitting).
    4. Fill categorical column NaNs with the string 'Unknown'.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (output of load_data).
    numerical_cols : list[str]
        Names of numerical columns to impute with median (default: from config).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering and splitting.
    """
    df = df.copy()

    # 1. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Derive length_of_stay if not already a column
    if "length_of_stay" not in df.columns:
        if "admission_date" in df.columns and "discharge_date" in df.columns:
            df["length_of_stay"] = (
                (df["discharge_date"] - df["admission_date"]).dt.total_seconds()
                / 86_400  # convert seconds → days
            ).round(2)

    # 3. Impute numerical columns with median
    for col in numerical_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # 4. Impute categorical columns with 'Unknown'
    categorical_cols_in_df = df.select_dtypes(
        include=["object", "category"]).columns
    for col in categorical_cols_in_df:
        df[col] = df[col].fillna("Unknown")

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
    X, y = separate_features_and_target(
        df,
        feature_columns=ALL_FEATURES,
        target_column=target_column,
        excluded_columns=EXCLUDED_COLUMNS,
        verbose=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test
