"""
run_baseline_comparison.py
---------------------------
Run baseline model comparison for the MEDILENS readmission prediction pipeline.

This script:
1. Loads and preprocesses data (same as main.py)
2. Trains both baseline and main model on training data
3. Evaluates both models on test data using identical metrics
4. Compares performance and generates reports

Usage
-----
python run_baseline_comparison.py

Output
------
- Console: Formatted comparison table
- reports/baseline_comparison.json: Detailed comparison report
- reports/baseline_vs_main_model.csv: Comparison table (CSV)
"""

import os
import pandas as pd

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    REPORTS_DIR,
)
from src.data_preprocessing import load_data, validate_schema, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns
from src.train import train_model
from src.baseline import train_baseline_model, get_baseline_description
from src.model_comparison import (
    compare_models,
    print_comparison_table,
    save_comparison_report,
    generate_comparison_dataframe,
)


def main():
    print("=" * 80)
    print("MEDILENS — BASELINE MODEL COMPARISON")
    print("=" * 80)
    print()

    # ── Step 1: Load and preprocess data ────────────────────────────────────
    print("[1/6] Loading and preprocessing data ...")
    df = load_data(DATA_PATH)
    validate_schema(df)
    df_clean = clean_data(df)
    print(f"      Loaded {df_clean.shape[0]:,} rows")

    # ── Step 2: Split data ──────────────────────────────────────────────────
    print(f"[2/6] Splitting data (test_size={TEST_SIZE}, random_state={RANDOM_STATE}) ...")
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    print(f"      Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # Get baseline description before preprocessing
    baseline_desc = get_baseline_description(y_train, strategy="most_frequent")
    print(f"\n      Baseline Strategy: {baseline_desc['strategy']}")
    print(f"      {baseline_desc['description']}")
    print(f"      Training class distribution: {baseline_desc['class_distribution']}")
    print(f"      Majority class: {baseline_desc['majority_class']} ({baseline_desc['majority_percentage']:.1f}%)")
    print(f"      Dataset is {'imbalanced' if baseline_desc['is_imbalanced'] else 'balanced'}")

    # ── Step 3: Preprocess features ─────────────────────────────────────────
    print("\n[3/6] Preprocessing features ...")
    X_train = drop_id_columns(X_train)
    X_test = drop_id_columns(X_test)

    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_proc = pipeline.fit_transform(X_train)  # fit on train only
    X_test_proc = pipeline.transform(X_test)  # transform test
    print(f"      Preprocessed shape: {X_train_proc.shape}")

    # ── Step 4: Train baseline model ────────────────────────────────────────
    print("\n[4/6] Training baseline model (DummyClassifier) ...")
    baseline_model = train_baseline_model(
        X_train_proc,
        y_train,
        strategy="most_frequent",
        random_state=RANDOM_STATE,
    )
    print("      Baseline model trained (always predicts majority class)")

    # ── Step 5: Train main model ────────────────────────────────────────────
    print("\n[5/6] Training main model (RandomForestClassifier) ...")
    main_model = train_model(X_train_proc, y_train, random_state=RANDOM_STATE)
    print("      Main model trained")

    # ── Step 6: Compare models ──────────────────────────────────────────────
    print("\n[6/6] Comparing models on test data ...")
    comparison = compare_models(baseline_model, main_model, X_test_proc, y_test)

    # Print comparison table
    print_comparison_table(comparison)

    # ── Save reports ────────────────────────────────────────────────────────
    print("Saving comparison reports ...")

    # Save JSON report
    json_path = os.path.join(REPORTS_DIR, "baseline_comparison.json")
    save_comparison_report(comparison, baseline_desc, json_path)
    print(f"  ✓ JSON report saved → {json_path}")

    # Save CSV table
    csv_path = os.path.join(REPORTS_DIR, "baseline_vs_main_model.csv")
    df_comparison = generate_comparison_dataframe(comparison)
    df_comparison.to_csv(csv_path)
    print(f"  ✓ CSV table saved  → {csv_path}")

    # ── Final assessment ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    summary = comparison["summary"]
    print(f"\n{summary['overall_improvement']}")
    print(f"Average improvement across all metrics: {summary['average_improvement']:.4f}")

    if summary['is_meaningful']:
        print("\n✓ The main model provides MEANINGFUL improvement over the baseline.")
        print("  The added complexity of the ML model is justified.")
    else:
        print("\n⚠ The main model shows MARGINAL improvement over the baseline.")
        print("  Consider:")
        print("  - Feature engineering to improve predictive signal")
        print("  - Hyperparameter tuning")
        print("  - Collecting more/better data")
        print("  - Using a different model architecture")

    # Special note for imbalanced datasets
    if baseline_desc['is_imbalanced']:
        print("\n⚠ Dataset is imbalanced. Key metrics to focus on:")
        print("  - Precision, Recall, F1-score (not just accuracy)")
        print("  - ROC-AUC (measures discrimination ability)")
        print("  - Per-class performance (especially minority class)")

    print("\n" + "=" * 80)
    print("Baseline comparison complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
