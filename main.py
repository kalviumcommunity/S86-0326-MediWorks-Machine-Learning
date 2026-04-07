"""
main.py
-------
MEDILENS — Hospital Readmission Prediction Pipeline
Orchestrates: load → clean → split → preprocess → train → evaluate → save → predict
"""

import json
import os
import pandas as pd

from src.config import (
    DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    CATEGORICAL_COLS, NUMERICAL_COLS, ID_COLUMNS,
    MODEL_PATH, PIPELINE_PATH, METRICS_REPORT_PATH,
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


def main():
    print("=== MEDILENS: Hospital Readmission Prediction ===\n")

    # 1. Load data
    df = load_data(DATA_PATH)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Validate schema
    validate_schema(df)
    print("Schema validated successfully")

    # 3. Clean data
    df_clean = clean_data(df)
    print(f"Cleaned data: {df_clean.shape[0]} rows remaining")

    # 4. Split data
    X_train, X_test, y_train, y_test = split_data(
        df_clean, target_column=TARGET_COLUMN,
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    # 4a. Validate feature separation (before dropping IDs)
    validate_feature_separation(X_train, y_train, target_column=TARGET_COLUMN)
    print("✓ Feature separation validated: target not in features, no excluded columns present")

    # 5. Drop ID columns
    X_train = drop_id_columns(X_train)
    X_test  = drop_id_columns(X_test)

    # 5a. Print feature summary
    print_feature_summary(X_train, NUMERICAL_COLS, CATEGORICAL_COLS)

    # 6. Preprocess (fit only on training data)
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_proc = pipeline.fit_transform(X_train)   # fit + transform on train
    X_test_proc  = pipeline.transform(X_test)         # transform only on test
    print(f"Preprocessed shape — Train: {X_train_proc.shape}, Test: {X_test_proc.shape}")

    # 7. Train model
    model = train_model(X_train_proc, y_train, random_state=RANDOM_STATE)
    print("Model training complete")

    # 8. Evaluate
    metrics = evaluate_model(model, X_test_proc, y_test)
    print("\n--- Evaluation Results ---")
    for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        print(f"  {k}: {metrics[k]:.4f}")

    # 9. Save artifacts
    save_artifacts(model, pipeline, MODEL_PATH, PIPELINE_PATH)
    print(f"\nModel saved    → {MODEL_PATH}")
    print(f"Pipeline saved → {PIPELINE_PATH}")

    # 10. Save metrics report
    os.makedirs(os.path.dirname(METRICS_REPORT_PATH), exist_ok=True)
    with open(METRICS_REPORT_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved  → {METRICS_REPORT_PATH}")

    # 11. Quick prediction demo
    sample = X_test.head(3).copy()
    result = predict(sample, model_path=MODEL_PATH, pipeline_path=PIPELINE_PATH)
    print("\n--- Sample Predictions ---")
    print(result[["predicted_readmission", "readmission_probability"]].to_string(index=False))

    print("\nPipeline complete!")


if __name__ == "__main__":
    main()
