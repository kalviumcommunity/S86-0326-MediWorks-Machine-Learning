"""
logistic_regression_tutorial.py
-------------------------------
Complete Logistic Regression workflow example for binary classification.

This script demonstrates the standard classification workflow:
1. Load and clean data
2. Split with stratification
3. Build majority-class baseline
4. Train Logistic Regression in a leakage-safe pipeline
5. Evaluate with accuracy, ROC-AUC, and class-wise metrics
6. Compare against baseline
7. Run cross-validation for stability
8. Interpret coefficients as odds ratios

Usage
-----
    python -m src.logistic_regression_tutorial

Output
------
Prints model comparison, CV stability summary, and interpretable coefficients.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
)
from src.data_preprocessing import load_data, validate_schema, clean_data
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns


def evaluate_classifier(name, y_true, y_pred, y_prob):
    """Return and print key classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    print(f"{name:28s} | Accuracy: {acc:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    return {"accuracy": acc, "f1": f1, "roc_auc": auc}


def coefficient_table_from_pipeline(pipeline):
    """Build a coefficient table with odds ratios for the fitted pipeline."""
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coef = model.coef_[0]
    odds_ratio = np.exp(coef)

    coef_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coef,
            "Odds Ratio": odds_ratio,
        }
    ).sort_values("Coefficient", key=np.abs, ascending=False)

    return coef_df


def run_workflow():
    """Execute the complete Logistic Regression tutorial workflow."""
    print("\n" + "=" * 80)
    print("LOGISTIC REGRESSION CLASSIFICATION WORKFLOW")
    print("=" * 80)

    # 1) Load and clean data
    print("\n[1/8] Loading and cleaning data")
    df = load_data(DATA_PATH)
    validate_schema(df)
    df = clean_data(df)
    print(f"  Samples: {len(df):,} | Columns: {df.shape[1]}")

    # 2) Prepare features and stratified split
    print("\n[2/8] Train/test split with stratification")
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X = drop_id_columns(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Positive rate (train): {y_train.mean():.3f}")
    print(f"  Positive rate (test):  {y_test.mean():.3f}")

    # 3) Baseline model
    print("\n[3/8] Training majority-class baseline")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_pred = baseline.predict(X_test)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]

    # 4) Logistic Regression pipeline
    print("\n[4/8] Training Logistic Regression pipeline")
    preprocessor = build_preprocessing_pipeline(
        CATEGORICAL_COLS, NUMERICAL_COLS)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    model_pred = pipeline.predict(X_test)
    model_prob = pipeline.predict_proba(X_test)[:, 1]

    # 5) Compare baseline vs model
    print("\n[5/8] Baseline comparison")
    baseline_metrics = evaluate_classifier(
        "Baseline (most_frequent)", y_test, baseline_pred, baseline_prob
    )
    model_metrics = evaluate_classifier(
        "Logistic Regression", y_test, model_pred, model_prob
    )

    print("\nDetailed classification report (Logistic Regression):")
    print(classification_report(y_test, model_pred, zero_division=0))

    # 6) Cross-validation stability
    print("\n[6/8] Cross-validation stability (5-fold)")
    cv_auc = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="roc_auc")
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1")
    print(f"  CV ROC-AUC: {cv_auc.round(3)}")
    print(f"  Mean ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
    print(f"  CV F1:      {cv_f1.round(3)}")
    print(f"  Mean F1:    {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

    # 7) Regularization strength tuning (C)
    print("\n[7/8] Hyperparameter tuning for regularization (C)")
    param_grid = {"model__C": [0.001, 0.01, 0.1, 1, 10, 100]}
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="roc_auc")
    grid.fit(X_train, y_train)
    print(f"  Best C: {grid.best_params_['model__C']}")
    print(f"  Best CV ROC-AUC: {grid.best_score_:.3f}")

    # 8) Coefficient interpretation
    print("\n[8/8] Coefficient interpretation (log-odds and odds ratios)")
    coef_df = coefficient_table_from_pipeline(pipeline)
    model_step = pipeline.named_steps["model"]
    print(f"  Intercept (log-odds): {model_step.intercept_[0]:.4f}")
    print("\nTop coefficients by absolute magnitude:")
    print(coef_df.head(20).to_string(index=False))

    # Practical checklist summary
    print("\n" + "=" * 80)
    print("PRACTICAL CHECKLIST")
    print("=" * 80)
    checks = [
        (
            "Model beats baseline on Accuracy",
            model_metrics["accuracy"] > baseline_metrics["accuracy"],
        ),
        (
            "Model beats baseline on ROC-AUC",
            model_metrics["roc_auc"] > baseline_metrics["roc_auc"],
        ),
        (
            "Model beats baseline on F1",
            model_metrics["f1"] > baseline_metrics["f1"],
        ),
        ("Cross-validation ROC-AUC is stable", cv_auc.std() < 0.10),
    ]
    for message, status in checks:
        symbol = "[OK]" if status else "[WARN]"
        print(f"  {symbol} {message}")

    print("\nTutorial complete.")
    print("Use this script as your reference implementation for Logistic Regression.")


if __name__ == "__main__":
    run_workflow()
