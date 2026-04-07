```python
"""
config.py
---------
Centralized configuration for the MEDILENS ML pipeline.
"""

import os

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR      = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR  = os.path.join(DATA_DIR, "raw")

MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR   = os.path.join(PROJECT_ROOT, "reports")
LOGS_DIR      = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(RAW_DATA_DIR, "hospital_visits.csv")

# ---------------------------------------------------------------------------
# Column Definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "readmitted"

# ID Columns
ID_COLUMNS = ["patient_id"]

# Numerical Features
NUMERICAL_FEATURES = [
    "age",
    "length_of_stay",
    "num_procedures",
    "num_medications",
    "num_diagnoses",
]

# Categorical Features
CATEGORICAL_FEATURES = [
    "department",
    "gender",
    "admission_type",
    "bed_type",
]

# Datetime Columns
DATETIME_COLS = ["admission_date", "discharge_date"]

# Excluded Columns (not used for modeling)
EXCLUDED_COLUMNS = [
    "patient_id",
    "admission_date",
    "discharge_date",
]

# ---------------------------------------------------------------------------
# Aliases (for compatibility across modules)
# ---------------------------------------------------------------------------

NUMERICAL_COLS   = NUMERICAL_FEATURES
CATEGORICAL_COLS = CATEGORICAL_FEATURES

# All model features
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ---------------------------------------------------------------------------
# Validation Checks
# ---------------------------------------------------------------------------

assert TARGET_COLUMN not in ALL_FEATURES, \
    "Target column must not be in features"

assert len(set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)) == 0, \
    "Numerical and categorical features must not overlap"

for col in EXCLUDED_COLUMNS:
    assert col not in ALL_FEATURES, \
        f"{col} should not be used in features"

# Required dataset columns
REQUIRED_COLUMNS = (
    ID_COLUMNS
    + NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
    + DATETIME_COLS
    + [TARGET_COLUMN]
)

# ---------------------------------------------------------------------------
# Split Settings
# ---------------------------------------------------------------------------

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model Parameters
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PATH    = os.path.join(MODELS_DIR, "random_forest_readmission.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

METRICS_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
PROBLEM_DEFINITION_REPORT_PATH = os.path.join(REPORTS_DIR, "problem_definition.json")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

EXPERIMENT_LOG_PATH = os.path.join(LOGS_DIR, "experiment_log.csv")
```
