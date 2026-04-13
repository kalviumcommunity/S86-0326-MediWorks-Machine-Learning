"""
f1_score_tutorial.py
--------------------
Hands-on F1-Score tutorial for imbalanced binary classification.

This script focuses on practical F1 usage:
1. Why harmonic mean is stricter than arithmetic mean
2. Why accuracy can mislead on imbalance
3. Baseline vs model comparison
4. Validation-safe threshold optimization for F1
5. Cross-validation with F1
6. Micro/Macro/Weighted F1 examples for multi-class settings

Usage
-----
    python -m src.f1_score_tutorial
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import (
    CATEGORICAL_COLS,
    DATA_PATH,
    NUMERICAL_COLS,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.data_preprocessing import clean_data, load_data, validate_schema
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns


def harmonic_mean(p: float, r: float) -> float:
    """Compute harmonic mean for Precision/Recall with safe zero handling."""
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)


def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    """Return key binary classification metrics at a chosen threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "y_pred": y_pred,
    }


def print_metric_row(label: str, scores: dict) -> None:
    """Pretty-print compact metric row."""
    print(
        f"{label:28s} | "
        f"Acc: {scores['accuracy']:.3f} | "
        f"P: {scores['precision']:.3f} | "
        f"R: {scores['recall']:.3f} | "
        f"F1: {scores['f1']:.3f}"
    )


def run_tutorial() -> None:
    """Run end-to-end F1 lesson with project data."""
    print("\n" + "=" * 80)
    print("F1-SCORE TUTORIAL: PRECISION-RECALL BALANCE")
    print("=" * 80)

    # 1) Harmonic vs arithmetic mean behavior
    print("\n[1/6] Harmonic mean intuition")
    p_ex, r_ex = 0.90, 0.10
    arithmetic = (p_ex + r_ex) / 2
    harmonic = harmonic_mean(p_ex, r_ex)
    print(f"  Precision={p_ex:.2f}, Recall={r_ex:.2f}")
    print(f"  Arithmetic mean: {arithmetic:.3f}")
    print(f"  Harmonic mean(F1): {harmonic:.3f}")

    # 2) Load and split data
    print("\n[2/6] Loading data and creating train/val/test splits")
    df = load_data(DATA_PATH)
    validate_schema(df)
    df = clean_data(df)

    X = drop_id_columns(df.drop(columns=[TARGET_COLUMN]))
    y = df[TARGET_COLUMN]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    print(
        f"  Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | "
        f"Test: {len(X_test):,}"
    )
    print(f"  Positive rate (test): {y_test.mean():.3f}")

    # 3) Baseline and model
    print("\n[3/6] Baseline vs Logistic Regression at default threshold=0.50")
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    base_prob = baseline.predict_proba(X_test)[:, 1]
    base_scores = metrics_at_threshold(y_test, base_prob, threshold=0.50)

    model = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS),
            ),
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
    model.fit(X_train, y_train)

    test_prob = model.predict_proba(X_test)[:, 1]
    default_scores = metrics_at_threshold(y_test, test_prob, threshold=0.50)

    print_metric_row("Baseline (most_frequent)", base_scores)
    print_metric_row("LogReg @ threshold 0.50", default_scores)

    # 4) Threshold optimization on validation (not test)
    print("\n[4/6] Threshold tuning for F1 using validation set")
    val_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.05)
    val_f1s = [
        f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    best_threshold = float(thresholds[int(np.argmax(val_f1s))])
    tuned_scores = metrics_at_threshold(
        y_test, test_prob, threshold=best_threshold)

    print(f"  Best validation threshold: {best_threshold:.2f}")
    print(f"  Best validation F1:        {max(val_f1s):.3f}")
    print_metric_row(f"LogReg @ threshold {best_threshold:.2f}", tuned_scores)

    print("\n  Classification report at tuned threshold:")
    print(classification_report(
        y_test, tuned_scores["y_pred"], zero_division=0))

    # 5) Cross-validation with F1
    print("\n[5/6] Cross-validation stability using F1")
    cv_f1 = cross_val_score(model, X_train_full,
                            y_train_full, cv=5, scoring="f1")
    print(f"  CV F1 scores: {np.round(cv_f1, 3)}")
    print(f"  Mean CV F1:   {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")

    # 6) Multi-class F1 averaging demo
    print("\n[6/6] Multi-class F1 averaging (micro/macro/weighted)")
    y_true_mc = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    y_pred_mc = np.array([0, 0, 1, 1, 0, 2, 2, 0, 2])

    f1_micro = f1_score(y_true_mc, y_pred_mc, average="micro")
    f1_macro = f1_score(y_true_mc, y_pred_mc, average="macro")
    f1_weighted = f1_score(y_true_mc, y_pred_mc, average="weighted")

    print(f"  Micro F1:    {f1_micro:.3f}")
    print(f"  Macro F1:    {f1_macro:.3f}")
    print(f"  Weighted F1: {f1_weighted:.3f}")

    print("\nKey takeaway: report Precision + Recall + F1 together, and always compare")
    print("against a baseline with threshold tuning done only on validation data.")


if __name__ == "__main__":
    run_tutorial()
