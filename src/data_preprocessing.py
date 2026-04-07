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
    REQUIRED_COLUMNS,
    DATETIME_COLS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERICAL_COLS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    EXCLUDED_COLUMNS,
    ALL_FEATURES,
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

def validate_schema(df: pd.DataFrame, required_columns: list[str] = REQUIRED_COLUMNS) -> None:
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
    categorical_cols_in_df = df.select_dtypes(include=["object", "category"]).columns
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


# ---------------------------------------------------------------------------
# 5. Feature Type Validation
# ---------------------------------------------------------------------------

def validate_feature_separation(
    X: pd.DataFrame,
    y: pd.Series,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Validate that features and target are properly separated.

    This function performs critical checks to ensure:
    1. Target column is not present in the feature DataFrame
    2. Excluded columns are not present in the feature DataFrame
    3. All expected features are present

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (after splitting, before preprocessing).
    y : pd.Series
        Target series.
    target_column : str
        Name of the target column (default: from config).

    Raises
    ------
    ValueError
        If target column is found in features or if excluded columns are present.
    """
    # Check 1: Target must not be in features
    if target_column in X.columns:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: Target column '{target_column}' "
            f"is present in feature DataFrame. This will cause overfitting."
        )

    # Check 2: Excluded columns must not be in features
    excluded_present = [col for col in EXCLUDED_COLUMNS if col in X.columns]
    if excluded_present:
        raise ValueError(
            f"INVALID FEATURES DETECTED: Excluded columns {excluded_present} "
            f"are present in feature DataFrame. These must be removed before modeling."
        )

    # Check 3: All expected features should be present (after ID removal)
    expected_features = set(ALL_FEATURES)
    actual_features = set(X.columns)
    
    # Account for ID columns that may have been dropped
    expected_features_after_id_drop = expected_features - set(EXCLUDED_COLUMNS)
    
    missing_features = expected_features_after_id_drop - actual_features
    if missing_features:
        print(
            f"WARNING: Expected features missing from DataFrame: {sorted(missing_features)}"
        )


def print_feature_summary(
    X: pd.DataFrame,
    numerical_features: list[str] = NUMERICAL_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
) -> None:
    """
    Print a summary of feature types for validation and documentation.

    This function provides a clear overview of how features are grouped,
    which is essential for:
    - Verifying correct feature type assignment
    - Documenting the preprocessing pipeline
    - Debugging preprocessing issues

    Parameters
    ----------
    X : pd.DataFrame
        Feature DataFrame (after ID removal, before preprocessing).
    numerical_features : list[str]
        List of numerical feature names (default: from config).
    categorical_features : list[str]
        List of categorical feature names (default: from config).
    """
    # Count features present in DataFrame
    num_present = [col for col in numerical_features if col in X.columns]
    cat_present = [col for col in categorical_features if col in X.columns]
    
    print("\n" + "="*60)
    print("FEATURE TYPE SUMMARY")
    print("="*60)
    print(f"Total features in DataFrame: {len(X.columns)}")
    print(f"\nNumerical features: {len(num_present)}")
    for col in num_present:
        print(f"  - {col}")
    print(f"\nCategorical features: {len(cat_present)}")
    for col in cat_present:
        print(f"  - {col}")
    print(f"\nTotal defined features: {len(num_present) + len(cat_present)}")
    print("="*60 + "\n")
