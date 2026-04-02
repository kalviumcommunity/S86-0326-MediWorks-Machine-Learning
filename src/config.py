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

DATA_DIR        = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR    = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
EXTERNAL_DIR    = os.path.join(DATA_DIR, "external")

MODELS_DIR      = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR     = os.path.join(PROJECT_ROOT, "reports")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "logs")
NOTEBOOKS_DIR   = os.path.join(PROJECT_ROOT, "notebooks")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(RAW_DATA_DIR, "hospital_visits.csv")

# ---------------------------------------------------------------------------
# Column Definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "readmitted"

ID_COLUMNS = ["patient_id"]

NUMERICAL_COLS = [
    "age",
    "length_of_stay",
    "num_procedures",
    "num_medications",
    "num_diagnoses",
]

CATEGORICAL_COLS = [
    "department",
    "gender",
    "admission_type",
    "bed_type",
]

DATETIME_COLS = ["admission_date", "discharge_date"]

REQUIRED_COLUMNS = (
    ID_COLUMNS
    + NUMERICAL_COLS
    + CATEGORICAL_COLS
    + DATETIME_COLS
    + [TARGET_COLUMN]
)

# ---------------------------------------------------------------------------
# Splitting / Reproducibility
# ---------------------------------------------------------------------------

TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Model Hyperparameters
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
# Artifact Paths
# ---------------------------------------------------------------------------

MODEL_PATH    = os.path.join(MODELS_DIR, "random_forest_readmission.pkl")
PIPELINE_PATH = os.path.join(MODELS_DIR, "preprocessing_pipeline.pkl")

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

METRICS_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_metrics.json")

# ---------------------------------------------------------------------------
# Experiment Logging
# ---------------------------------------------------------------------------

EXPERIMENT_LOG_PATH = os.path.join(LOGS_DIR, "experiment_log.csv")