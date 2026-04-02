"""
config.py
---------
Centralized configuration for the MEDILENS ML pipeline.

All file paths, model hyperparameters, column names, and constants live here.
Functions across the pipeline import from this module, avoiding hardcoded values
and making the project portable and easy to maintain.
"""

import os

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------

# Project root is one level above src/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR           = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR      = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR  = os.path.join(DATA_DIR, "external")
MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR        = os.path.join(PROJECT_ROOT, "reports")
LOGS_DIR           = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# Place your hospital visit CSV file here before running the pipeline
DATA_PATH = os.path.join(RAW_DATA_DIR, "hospital_visits.csv")

# ---------------------------------------------------------------------------
# Column Definitions
# ---------------------------------------------------------------------------

TARGET_COLUMN = "readmitted"          # Binary target: 1 = readmitted within 30 days

# Columns that uniquely identify a patient visit — dropped before modelling
ID_COLUMNS = ["patient_id"]

# Columns used as-is (numeric) for the model
NUMERICAL_COLS = [
    "age",
    "length_of_stay",
    "num_procedures",
    "num_medications",
    "num_diagnoses",
]

# Columns that need one-hot encoding
CATEGORICAL_COLS = [
    "department",
    "gender",
    "admission_type",
    "bed_type",
]

# Datetime columns parsed during data ingestion
DATETIME_COLS = ["admission_date", "discharge_date"]

# Required columns the dataset MUST contain
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

TEST_SIZE     = 0.20   # 20 % held out for evaluation
RANDOM_STATE  = 42     # Global seed — controls all randomness in the pipeline

# ---------------------------------------------------------------------------
# Model Hyperparameters
# ---------------------------------------------------------------------------

MODEL_PARAMS = {
    "n_estimators":  200,
    "max_depth":     10,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "class_weight":  "balanced",   # handles class imbalance in readmission data
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,           # use all available CPU cores
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
