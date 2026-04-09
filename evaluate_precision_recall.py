"""
evaluate_precision_recall.py
-----------------------------
Assignment: Evaluating Classification Models Using Precision and Recall

Covers:
- Train/test split (no leakage)
- Majority-class baseline (DummyClassifier)
- Precision, Recall, Accuracy, Confusion Matrix
- Baseline vs Model comparison
- Threshold adjustment and PR curve
- F1 and F2 scores
- Cross-validation for Precision and Recall
- Business interpretation for readmission prediction

Usage:
    python evaluate_precision_recall.py
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
)

from src.config import (
    DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
    CATEGORICAL_COLS, NUMERICAL_COLS, MODEL_PARAMS, REPORTS_DIR,
)
from src.data_preprocessing import load_data, validate_schema, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sep(title="", char="="):
    width = 80
    print(f"\n{char * width}")
    if title:
        print(f" {title}")
        print(char * width)
    print()


def print_confusion_matrix(cm: np.ndarray):
    tn, fp, fn, tp = cm.ravel()
    total = cm.sum()
    print("  Confusion Matrix:")
    print(f"                    Predicted 0   Predicted 1")
    print(f"  Actual 0 (neg)       {tn:5d}         {fp:5d}   (TN / FP)")
    print(f"  Actual 1 (pos)       {fn:5d}         {tp:5d}   (FN / TP)")
    print()
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}  Total={total}")
    print(f"  False Negatives (missed readmissions) : {fn}  — most costly error")
    print(f"  False Positives (unnecessary alerts)  : {fp}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep("MEDILENS — Precision & Recall Evaluation")

    # ── 1. Load & clean ─────────────────────────────────────────────────────
    sep("1. Load Data", "-")
    df = load_data(DATA_PATH)
    validate_schema(df)
    df = clean_data(df)
    print(f"  Rows: {len(df):,}  |  Target: {TARGET_COLUMN}")

    dist = df[TARGET_COLUMN].value_counts()
    total = len(df)
    for cls, cnt in sorted(dist.items()):
        print(f"  Class {cls}: {cnt} ({cnt/total*100:.1f}%)")

    # ── 2. Stratified split ──────────────────────────────────────────────────
    sep("2. Stratified Train/Test Split (80/20)", "-")
    X_train, X_test, y_train, y_test = split_data(
        df, target_column=TARGET_COLUMN,
        test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Train class 1 rate: {y_train.mean()*100:.1f}%")
    print(f"  Test  class 1 rate: {y_test.mean()*100:.1f}%")

    # ── 3. Preprocess ────────────────────────────────────────────────────────
    sep("3. Preprocessing (fit on train only)", "-")
    X_train = drop_id_columns(X_train)
    X_test  = drop_id_columns(X_test)
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc  = pipeline.transform(X_test)
    print(f"  Feature matrix shape: {X_train_proc.shape}")

    # ── 4. Baseline ──────────────────────────────────────────────────────────
    sep("4. Majority-Class Baseline (DummyClassifier)", "-")
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_train_proc, y_train)
    b_pred = baseline.predict(X_test_proc)

    b_acc  = accuracy_score(y_test, b_pred)
    b_prec = precision_score(y_test, b_pred, zero_division=0)
    b_rec  = recall_score(y_test, b_pred, zero_division=0)
    b_f1   = f1_score(y_test, b_pred, zero_division=0)

    print(f"  Accuracy : {b_acc:.4f}")
    print(f"  Precision: {b_prec:.4f}")
    print(f"  Recall   : {b_rec:.4f}")
    print(f"  F1-Score : {b_f1:.4f}")
    print()
    print_confusion_matrix(confusion_matrix(y_test, b_pred))
    print()
    print("  Classification Report (Baseline):")
    print(classification_report(y_test, b_pred,
          target_names=["Not Readmitted", "Readmitted"], zero_division=0))

    # ── 5. Train model ───────────────────────────────────────────────────────
    sep("5. Train RandomForestClassifier", "-")
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train_proc, y_train)
    m_pred = model.predict(X_test_proc)
    m_prob = model.predict_proba(X_test_proc)[:, 1]
    print("  Model trained.")

    # ── 6. Evaluate model ────────────────────────────────────────────────────
    sep("6. Model Evaluation on Test Set", "-")
    m_acc  = accuracy_score(y_test, m_pred)
    m_prec = precision_score(y_test, m_pred, zero_division=0)
    m_rec  = recall_score(y_test, m_pred, zero_division=0)
    m_f1   = f1_score(y_test, m_pred, zero_division=0)
    m_f2   = fbeta_score(y_test, m_pred, beta=2, zero_division=0)

    print(f"  Accuracy : {m_acc:.4f}")
    print(f"  Precision: {m_prec:.4f}")
    print(f"  Recall   : {m_rec:.4f}")
    print(f"  F1-Score : {m_f1:.4f}")
    print(f"  F2-Score : {m_f2:.4f}  (recall-weighted — relevant for readmission)")
    print()
    print_confusion_matrix(confusion_matrix(y_test, m_pred))
    print()
    print("  Classification Report (Model):")
    print(classification_report(y_test, m_pred,
          target_names=["Not Readmitted", "Readmitted"], zero_division=0))

    # ── 7. Baseline vs Model comparison ─────────────────────────────────────
    sep("7. Baseline vs Model Comparison", "-")
    comp = pd.DataFrame({
        "Metric"    : ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Baseline"  : [b_acc, b_prec, b_rec, b_f1],
        "Model"     : [m_acc, m_prec, m_rec, m_f1],
        "Δ"         : [m_acc-b_acc, m_prec-b_prec, m_rec-b_rec, m_f1-b_f1],
    })
    print(comp.to_string(index=False))

    # ── 8. Threshold adjustment ──────────────────────────────────────────────
    sep("8. Threshold Adjustment", "-")
    print("  Default threshold = 0.5")
    for thresh in [0.5, 0.4, 0.3, 0.2]:
        y_t = (m_prob >= thresh).astype(int)
        p = precision_score(y_test, y_t, zero_division=0)
        r = recall_score(y_test, y_t, zero_division=0)
        f = f1_score(y_test, y_t, zero_division=0)
        print(f"  Threshold={thresh:.1f}  Precision={p:.4f}  Recall={r:.4f}  F1={f:.4f}")

    # Find threshold for ≥60% recall
    precisions_curve, recalls_curve, thresholds_curve = precision_recall_curve(y_test, m_prob)
    target_recall = 0.60
    valid = np.where(recalls_curve[:-1] >= target_recall)[0]
    if len(valid) > 0:
        best_idx = valid[np.argmax(precisions_curve[valid])]
        best_thresh = thresholds_curve[best_idx]
        best_prec   = precisions_curve[best_idx]
        print(f"\n  Best threshold for ≥{target_recall:.0%} recall: {best_thresh:.3f}")
        print(f"  Precision at that threshold: {best_prec:.4f}")

    # ── 9. PR Curve ──────────────────────────────────────────────────────────
    sep("9. Precision-Recall Curve", "-")
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recalls_curve, precisions_curve, marker=".", label="RandomForest")
    ax.axhline(y=m_prec, color="gray", linestyle="--", alpha=0.5, label=f"Default threshold precision={m_prec:.2f}")
    ax.axvline(x=m_rec,  color="gray", linestyle=":",  alpha=0.5, label=f"Default threshold recall={m_rec:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — Readmission Prediction")
    ax.legend()
    ax.grid(True)
    pr_curve_path = os.path.join(REPORTS_DIR, "precision_recall_curve.png")
    fig.savefig(pr_curve_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  PR curve saved → {pr_curve_path}")

    # ── 10. Cross-validation ─────────────────────────────────────────────────
    sep("10. Cross-Validation (5-fold Stratified)", "-")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_model = RandomForestClassifier(**MODEL_PARAMS)

    cv_prec = cross_val_score(cv_model, X_train_proc, y_train, cv=skf, scoring="precision")
    cv_rec  = cross_val_score(cv_model, X_train_proc, y_train, cv=skf, scoring="recall")
    cv_f1   = cross_val_score(cv_model, X_train_proc, y_train, cv=skf, scoring="f1")

    print(f"  CV Precision: {cv_prec.mean():.4f} ± {cv_prec.std():.4f}")
    print(f"  CV Recall   : {cv_rec.mean():.4f} ± {cv_rec.std():.4f}")
    print(f"  CV F1       : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # ── 11. Interpretation ───────────────────────────────────────────────────
    sep("11. Interpretation", "-")
    print("""
  WHICH METRIC MATTERS MORE IN THIS PROJECT?
  -------------------------------------------
  For hospital readmission prediction, RECALL is the priority metric.

  WHY?
  - A False Negative = patient predicted as "not readmitted" but actually readmitted.
    This means no early intervention, potential deterioration, and higher cost.
  - A False Positive = patient flagged for follow-up unnecessarily.
    This is a minor operational cost (extra check-in call or appointment).

  COST OF ERRORS:
  - False Negative (missed readmission): High — patient harm + hospital penalty
  - False Positive (unnecessary follow-up): Low — minor resource cost

  CONCLUSION:
  - Recall must be maximized first (catch as many at-risk patients as possible).
  - Precision is secondary (reduce unnecessary follow-ups where possible).
  - F2-score is the recommended single metric (weights recall 2x over precision).
  - Threshold should be lowered below 0.5 to improve recall at acceptable precision.

  FRAUD DETECTION SCENARIO (Video Answer):
  - False Negative cost: ₹50,000 per missed fraud case
  - False Positive cost: ₹1,000 per false alarm
  - Ratio: FN is 50x more costly than FP
  - → RECALL must be prioritized heavily
  - → Lower the threshold to catch more fraud, accepting more false alarms
  - → Use F2-score or set minimum recall threshold (e.g., ≥80%) then optimize precision
  - → Accuracy is irrelevant — 98% accuracy with 0% recall is a complete failure
    """)

    # ── 12. Save report ──────────────────────────────────────────────────────
    sep("12. Save Report", "-")
    report = {
        "dataset": {"rows": len(df), "class_distribution": dist.to_dict()},
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "baseline": {"accuracy": b_acc, "precision": b_prec, "recall": b_rec, "f1": b_f1},
        "model": {
            "accuracy": m_acc, "precision": m_prec,
            "recall": m_rec, "f1": m_f1, "f2": m_f2,
            "confusion_matrix": confusion_matrix(y_test, m_pred).tolist(),
        },
        "cross_validation": {
            "precision": {"mean": cv_prec.mean(), "std": cv_prec.std()},
            "recall":    {"mean": cv_rec.mean(),  "std": cv_rec.std()},
            "f1":        {"mean": cv_f1.mean(),   "std": cv_f1.std()},
        },
        "interpretation": {
            "priority_metric": "Recall",
            "reason": "False negatives (missed readmissions) are more costly than false positives",
            "recommended_threshold": "< 0.5 to improve recall",
            "recommended_single_metric": "F2-score",
        },
    }

    report_path = os.path.join(REPORTS_DIR, "precision_recall_evaluation.json")
    with open(report_path, "w") as f:
        json.dump({k: (v if not isinstance(v, (np.integer, np.floating)) else float(v))
                   for k, v in report.items()}, f, indent=4, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))
    print(f"  Report saved → {report_path}")

    sep("EVALUATION COMPLETE")
    print(f"  Model Precision : {m_prec:.4f}  (Baseline: {b_prec:.4f})")
    print(f"  Model Recall    : {m_rec:.4f}  (Baseline: {b_rec:.4f})")
    print(f"  Model F1        : {m_f1:.4f}  (Baseline: {b_f1:.4f})")
    print(f"  Model F2        : {m_f2:.4f}")
    print(f"  Priority metric : Recall (readmission detection)")
    print()


if __name__ == "__main__":
    main()
