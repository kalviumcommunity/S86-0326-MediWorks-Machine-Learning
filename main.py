```python
"""
main.py
-------
MEDILENS — Hospital Readmission Prediction Pipeline
Entry point that orchestrates the complete ML workflow:

    load → validate → clean → split → preprocess → train → evaluate → save → log → predict

Run
---
    python main.py
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
    MODEL_PATH,
    PIPELINE_PATH,
    METRICS_REPORT_PATH,
    MODELS_DIR,
    REPORTS_DIR,
    PROBLEM_DEFINITION_REPORT_PATH,
)
from src.data_preprocessing import (
    load_data, validate_schema, clean_data, split_data,
    validate_feature_separation, print_feature_summary
)
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns
from src.train import train_model
from src.evaluate import evaluate_model
from src.persistence import save_artifacts
from src.predict import predict
from src.problem_definition import infer_supervised_problem_type


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
EXPERIMENT_LOG = os.path.join(LOGS_DIR, "experiment_log.csv")

LOG_FIELDS = [
    "timestamp", "algorithm", "test_size", "random_state",
    "accuracy", "precision", "recall", "f1", "roc_auc",
]


def log_experiment(metrics: dict) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    write_header = not os.path.exists(EXPERIMENT_LOG)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "RandomForestClassifier",
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
    }

    with open(EXPERIMENT_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    print("\nMEDILENS PIPELINE STARTED\n")

    # Step 1: Load
    df = load_data(DATA_PATH)

    # Step 2: Validate
    validate_schema(df)

    # Step 3: Clean
    df_clean = clean_data(df)

    # Step 4: Problem Type
    problem_definition = infer_supervised_problem_type(df_clean[TARGET_COLUMN])
    print(problem_definition)

    # Step 5: Split
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Step 6: Feature Engineering + Validation
    validate_feature_separation(X_train, y_train, target_column=TARGET_COLUMN)

    X_train = drop_id_columns(X_train)
    X_test  = drop_id_columns(X_test)

    print_feature_summary(X_train, NUMERICAL_COLS, CATEGORICAL_COLS)

    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)

    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc  = pipeline.transform(X_test)

    # Step 7: Train
    model = train_model(X_train_proc, y_train, random_state=RANDOM_STATE)

    # Step 8: Evaluate
    metrics = evaluate_model(model, X_test_proc, y_test)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Step 9: Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)

    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(PROBLEM_DEFINITION_REPORT_PATH, "w") as f:
        json.dump(problem_definition, f, indent=4)

    log_experiment(metrics)

    # Step 10: Predict sample
    sample = X_test.head(3).copy()
    result = predict(sample, model_path=MODEL_PATH, pipeline_path=PIPELINE_PATH)

    print("\nSample Predictions:")
    print(result)

    print("\nPIPELINE COMPLETED\n")


if __name__ == "__main__":
    main()
```
