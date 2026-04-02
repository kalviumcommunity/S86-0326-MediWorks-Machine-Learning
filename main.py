"""
main.py
-------
MEDILENS — Hospital Readmission Prediction Pipeline
Entry point that orchestrates the complete ML workflow:

    load → validate → clean → split → preprocess → train → evaluate → save → log → predict

Run
---
    python main.py

Requires data/raw/hospital_visits.csv to exist.
Generate synthetic data first if needed:
    python generate_sample_dataset.py
"""

import csv
import json
import os
from datetime import datetime

import pandas as pd

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    ID_COLUMNS,
    MODEL_PATH,
    PIPELINE_PATH,
    METRICS_REPORT_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data_preprocessing import load_data, validate_schema, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts
from src.predict import predict

# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

LOGS_DIR      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
EXPERIMENT_LOG = os.path.join(LOGS_DIR, "experiment_log.csv")

LOG_FIELDS = [
    "timestamp", "algorithm", "n_estimators", "max_depth",
    "test_size", "random_state",
    "accuracy", "precision", "recall", "f1", "roc_auc",
]


def log_experiment(metrics: dict, model_params: dict) -> None:
    """Append one row to logs/experiment_log.csv."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    write_header = not os.path.exists(EXPERIMENT_LOG)

    row = {
        "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":    "RandomForestClassifier",
        "n_estimators": model_params.get("n_estimators", ""),
        "max_depth":    model_params.get("max_depth", ""),
        "test_size":    TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy":     metrics["accuracy"],
        "precision":    metrics["precision"],
        "recall":       metrics["recall"],
        "f1":           metrics["f1"],
        "roc_auc":      metrics["roc_auc"],
    }

    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Pipeline banner
# ---------------------------------------------------------------------------

BANNER = """
============================================================
  MEDILENS — Hospital Readmission Prediction Pipeline
============================================================
"""

SEP = "  " + "─" * 56


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print(BANNER)

    # ── Step 1: Load ────────────────────────────────────────
    print("[1/8] Loading raw data ...")
    df = load_data(DATA_PATH)
    print(f"      Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")

    # ── Step 2: Validate ────────────────────────────────────
    print("[2/8] Validating schema ...")
    validate_schema(df)
    print("      Schema OK — all required columns present.")

    # ── Step 3: Clean ───────────────────────────────────────
    print("[3/8] Cleaning data ...")
    df_clean = clean_data(df)
    print(f"      {df_clean.shape[0]:,} rows after deduplication and imputation.")

    # ── Step 4: Split ───────────────────────────────────────
    print(f"[4/8] Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE}) ...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"      Training rows : {len(X_train):,}")
    print(f"      Test rows     : {len(X_test):,}")

    # ── Step 5: Feature engineering ─────────────────────────
    print("[5/8] Dropping ID columns and building preprocessing pipeline ...")
    X_train = drop_id_columns(X_train)
    X_test  = drop_id_columns(X_test)

    pipeline     = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_proc = pipeline.fit_transform(X_train)   # fit + transform on train only
    X_test_proc  = pipeline.transform(X_test)         # transform only on test
    print(f"      Preprocessed shape — Train: {X_train_proc.shape}, Test: {X_test_proc.shape}")

    # ── Step 6: Train ───────────────────────────────────────
    print("[6/8] Training Random Forest model ...")
    from src.config import MODEL_PARAMS
    model = train_model(X_train_proc, y_train, random_state=RANDOM_STATE)
    print("      Training complete.")

    # ── Step 7: Evaluate ────────────────────────────────────
    print("[7/8] Evaluating model on held-out test set ...")
    metrics = evaluate_model(model, X_test_proc, y_test)
    print()
    print("  ── Evaluation Results ──────────────────────────────────")
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {key:<12}: {metrics[key]:.4f}")
    print(SEP)

    # ── Step 8: Save artifacts, metrics report, and log ─────
    print("[8/8] Saving artifacts and logging experiment ...")

    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
    print(f"  Model saved    → {MODEL_PATH}")
    print(f"  Pipeline saved → {PIPELINE_PATH}")

    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Metrics report → {METRICS_REPORT_PATH}")

    log_experiment(metrics, MODEL_PARAMS)
    print(f"  Experiment log → {EXPERIMENT_LOG}")

    # ── Quick prediction demo ────────────────────────────────
    print()
    sample = X_test.head(3).copy()
    result = predict(sample, model_path=MODEL_PATH, pipeline_path=PIPELINE_PATH)
    print("  ── Sample Predictions ──────────────────────────────────")
    print(result[["predicted_readmission", "readmission_probability"]].to_string(index=False))
    print(SEP)
    print()
    print("  Pipeline complete. MEDILENS model is ready.")
    print()


if __name__ == "__main__":
    main()
