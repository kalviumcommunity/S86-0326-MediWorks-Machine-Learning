"""
config.py
---------
Centralized configuration for the MEDILENS ML pipeline.

All file paths, model hyperparameters, column names, and constants live here.
"""

import os

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------

# Project root is one level above src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(RAW_DATA_DIR, "hospital_visits.csv")

# ---------------------------------------------------------------------------
# Column Definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "readmitted"  # Binary target: 1 = readmitted within 30 days

# --- Feature / target definitions (lesson requirement) ---

# Excluded columns with reasons
EXCLUDED_COLUMNS = [
    "patient_id",      # Identifier (no generalizable predictive signal)
    "admission_date",  # Raw timestamp (used only to derive length_of_stay)
    "discharge_date",  # Raw timestamp (used only to derive length_of_stay)
]

# Numerical features
NUMERICAL_FEATURES = [
    "age",
    "length_of_stay",
    "num_procedures",
    "num_medications",
    "num_diagnoses",
]

# Categorical features
CATEGORICAL_FEATURES = [
    "department",
    "gender",
    "admission_type",
    "bed_type",
]

# Derived
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Safety checks (fail fast if misconfigured)
if TARGET_COLUMN in ALL_FEATURES:
    raise ValueError(
        "TARGET_COLUMN must not appear in ALL_FEATURES (data leakage risk)")

_overlap = set(ALL_FEATURES).intersection(set(EXCLUDED_COLUMNS))
if _overlap:
    raise ValueError(
        f"ALL_FEATURES overlaps EXCLUDED_COLUMNS: {sorted(_overlap)}")

# --- Backwards-compatible aliases used elsewhere in the codebase ---
ID_COLUMNS = ["patient_id"]
DATETIME_COLS = ["admission_date", "discharge_date"]
NUMERICAL_COLS = NUMERICAL_FEATURES
CATEGORICAL_COLS = CATEGORICAL_FEATURES

# Columns the dataset MUST contain to run the pipeline.
# Note: `length_of_stay` can be derived from admission/discharge timestamps.
REQUIRED_CORE_COLUMNS = (
    ID_COLUMNS
    + ["age", "num_procedures", "num_medications", "num_diagnoses"]
    + CATEGORICAL_COLS
    + [TARGET_COLUMN]
)

# ---------------------------------------------------------------------------
# Splitting / Reproducibility
# ---------------------------------------------------------------------------

TEST_SIZE = 0.20   # 20 % held out for evaluation
RANDOM_STATE = 42     # Global seed — controls all randomness in the pipeline

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "class_weight":  "balanced",   # handles class imbalance in readmission data
    "random_state":  RANDOM_STATE,
    "n_jobs": -1,           # use all available CPU cores
}

# ---------------------------------------------------------------------------
# Artifact Paths
# ---------------------------------------------------------------------------

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_readmission.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRICS_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
PROBLEM_DEFINITION_REPORT_PATH = os.path.join(REPORTS_DIR, "problem_definition.json")

# ---------------------------------------------------------------------------
# Experiment Logging
# ---------------------------------------------------------------------------

EXPERIMENT_LOG_PATH = os.path.join(LOGS_DIR, "experiment_log.csv")