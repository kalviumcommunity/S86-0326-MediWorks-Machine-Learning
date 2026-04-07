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

# ---------------------------------------------------------------------------
# Feature Type Definitions
# ---------------------------------------------------------------------------
# This section explicitly defines which features are numerical and which are
# categorical based on conceptual reasoning, not automatic dtype detection.
#
# Design Principle:
# Feature types are determined by how the model should interpret them, not by
# how they are stored in the dataset. An integer column can be categorical.
# A string column may represent ordinal structure. Binary columns require
# careful consideration of their semantic meaning.
# ---------------------------------------------------------------------------

# NUMERICAL FEATURES
# ------------------
# Features that represent continuous or discrete quantities where arithmetic
# operations (mean, distance, scaling) are meaningful. These will be scaled
# using StandardScaler to ensure all features contribute equally to distance-
# based calculations and gradient descent.
#
# Justification:
# - age: Continuous quantity (years). Older patients may have different
#   readmission risk. Scaling ensures age doesn't dominate due to its range.
# - length_of_stay: Continuous quantity (days). Longer stays may indicate
#   severity and correlate with readmission. Requires scaling.
# - num_procedures: Discrete count. More procedures may indicate complexity.
#   Treated as numerical because counts are ordered and distances are meaningful.
# - num_medications: Discrete count. Higher medication count may indicate
#   chronic conditions. Treated as numerical for the same reason.
# - num_diagnoses: Discrete count. More diagnoses suggest comorbidity.
#   Treated as numerical because the magnitude matters.
#
NUMERICAL_FEATURES = [
    "age",
    "length_of_stay",
    "num_procedures",
    "num_medications",
    "num_diagnoses",
]

# CATEGORICAL FEATURES
# --------------------
# Features that represent discrete groups or labels where arithmetic operations
# are not meaningful. These will be one-hot encoded to create binary indicator
# variables for each category.
#
# Justification:
# - department: Nominal categorical. No inherent order between Emergency, ICU,
#   Surgery, etc. Each department may have different readmission patterns.
#   One-hot encoding creates separate binary features for each department.
# - gender: Nominal categorical. Male, Female, Other have no natural ordering.
#   Gender may correlate with certain health outcomes. One-hot encoded.
# - admission_type: Nominal categorical. Emergency, Elective, Urgent represent
#   different admission contexts with no inherent order. One-hot encoded.
# - bed_type: Nominal categorical. General, ICU, Private beds represent
#   different care levels but are not strictly ordered. One-hot encoded.
#
# Note on Ordinality:
# While some features (e.g., bed_type: General < ICU in terms of care intensity)
# could be argued as ordinal, we treat them as nominal because:
# 1. The ordering is not universally agreed upon
# 2. One-hot encoding is safer and lets the model learn relationships
# 3. Tree-based models (Random Forest) handle one-hot encoding well
#
CATEGORICAL_FEATURES = [
    "department",
    "gender",
    "admission_type",
    "bed_type",
]

# EXCLUDED COLUMNS
# ----------------
# Columns that must be removed before model training to prevent data leakage,
# avoid using non-predictive identifiers, or exclude features that are not
# available at prediction time.
#
# Justification:
# - patient_id: Unique identifier with no predictive value. Including it would
#   cause the model to memorize individual patients rather than learn patterns.
# - admission_date: Timestamp. While temporal patterns exist, the raw date is
#   not useful. Could be engineered into features (day_of_week, month) but is
#   excluded in the current pipeline to keep the model simple.
# - discharge_date: Timestamp. Not available at admission time (prediction time).
#   Including it would cause severe data leakage since discharge date is only
#   known after the visit ends. Must be excluded.
#
# Note on length_of_stay:
# length_of_stay is derived from admission_date and discharge_date but is
# included as a numerical feature because it represents the duration of care,
# which is a valid predictor of readmission risk. In a real-world deployment,
# this feature would need to be handled carefully (e.g., predicted or estimated
# at admission time, or the model would only be used post-discharge).
#
EXCLUDED_COLUMNS = [
    "patient_id",
    "admission_date",
    "discharge_date",
]

# LEGACY ALIASES (for backward compatibility)
# --------------------------------------------
ID_COLUMNS = ["patient_id"]
NUMERICAL_COLS = NUMERICAL_FEATURES
CATEGORICAL_COLS = CATEGORICAL_FEATURES
DATETIME_COLS = ["admission_date", "discharge_date"]

# ALL FEATURES (for validation)
# ------------------------------
# This list contains all features that will be used for modeling.
# It explicitly excludes the target column and excluded columns.
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# VALIDATION: Ensure target is not in features
assert TARGET_COLUMN not in ALL_FEATURES, \
    f"Target column '{TARGET_COLUMN}' must not be in feature list"

# VALIDATION: Ensure no overlap between numerical and categorical
assert len(set(NUMERICAL_FEATURES) & set(CATEGORICAL_FEATURES)) == 0, \
    "Numerical and categorical features must not overlap"

# VALIDATION: Ensure excluded columns are not in features
for col in EXCLUDED_COLUMNS:
    assert col not in ALL_FEATURES, \
        f"Excluded column '{col}' must not be in feature list"

# Required columns the dataset MUST contain
REQUIRED_COLUMNS = (
    ID_COLUMNS
    + NUMERICAL_FEATURES
    + CATEGORICAL_FEATURES
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
