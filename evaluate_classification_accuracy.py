"""
evaluate_classification_accuracy.py
------------------------------------
Assignment: Evaluating Classification Models Using Accuracy

This script implements a comprehensive evaluation of the MEDILENS readmission
prediction model, comparing it against a majority-class baseline and analyzing
whether Accuracy is a reliable metric for this dataset.

Assignment Requirements:
1. Stratified train-test split
2. Majority-class baseline using DummyClassifier
3. Model training and Accuracy computation
4. Confusion matrix generation and analysis
5. Additional metrics: Precision, Recall, F1-score, Balanced Accuracy
6. Cross-validation with mean ± std
7. Written interpretation of results

Usage:
    python evaluate_classification_accuracy.py
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    MODEL_PARAMS,
    REPORTS_DIR,
)
from src.data_preprocessing import load_data, validate_schema, clean_data, split_data
from src.feature_engineering import build_preprocessing_pipeline, drop_id_columns


# ============================================================================
# Helper Functions
# ============================================================================

def print_section_header(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 80}")
    print(f"{title.center(80)}")
    print(f"{char * 80}\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")


def format_metric(value: float, decimals: int = 4) -> str:
    """Format a metric value with specified decimal places."""
    return f"{value:.{decimals}f}"


def print_confusion_matrix_analysis(cm: np.ndarray, labels: list = [0, 1]) -> None:
    """
    Print detailed confusion matrix analysis.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix [[TN, FP], [FN, TP]]
    labels : list
        Class labels (default: [0, 1] for binary classification)
    """
    tn, fp, fn, tp = cm.ravel()
    total = tn + fp + fn + tp
    
    print("\nConfusion Matrix:")
    print(f"                 Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {tn:6d}              {fp:6d}")
    print(f"Actual Positive        {fn:6d}              {tp:6d}")
    
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives  (TN): {tn:5d} ({tn/total*100:5.2f}%) - Correctly predicted as NOT readmitted")
    print(f"  False Positives (FP): {fp:5d} ({fp/total*100:5.2f}%) - Incorrectly predicted as readmitted")
    print(f"  False Negatives (FN): {fn:5d} ({fn/total*100:5.2f}%) - Missed readmissions (Type II error)")
    print(f"  True Positives  (TP): {tp:5d} ({tp/total*100:5.2f}%) - Correctly predicted as readmitted")
    print(f"  Total samples:        {total:5d}")


def calculate_cross_validation_scores(
    model,
    X: np.ndarray,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = RANDOM_STATE
) -> dict:
    """
    Perform stratified k-fold cross-validation and return scores.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to evaluate
    X : np.ndarray
        Feature matrix
    y : pd.Series
        Target labels
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary containing mean and std for each metric
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    # Calculate cross-validation scores for multiple metrics
    cv_accuracy = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_precision = cross_val_score(model, X, y, cv=skf, scoring='precision')
    cv_recall = cross_val_score(model, X, y, cv=skf, scoring='recall')
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')
    
    return {
        'accuracy': {
            'mean': cv_accuracy.mean(),
            'std': cv_accuracy.std(),
            'scores': cv_accuracy.tolist()
        },
        'precision': {
            'mean': cv_precision.mean(),
            'std': cv_precision.std(),
            'scores': cv_precision.tolist()
        },
        'recall': {
            'mean': cv_recall.mean(),
            'std': cv_recall.std(),
            'scores': cv_recall.tolist()
        },
        'f1': {
            'mean': cv_f1.mean(),
            'std': cv_f1.std(),
            'scores': cv_f1.tolist()
        }
    }


def interpret_results(
    baseline_metrics: dict,
    model_metrics: dict,
    class_distribution: dict,
    cv_results: dict
) -> dict:
    """
    Generate comprehensive interpretation of evaluation results.
    
    Parameters
    ----------
    baseline_metrics : dict
        Metrics from baseline model
    model_metrics : dict
        Metrics from trained model
    class_distribution : dict
        Training class distribution
    cv_results : dict
        Cross-validation results
        
    Returns
    -------
    dict
        Interpretation dictionary with key findings
    """
    interpretation = {}
    
    # 1. Baseline comparison
    acc_improvement = model_metrics['accuracy'] - baseline_metrics['accuracy']
    f1_improvement = model_metrics['f1'] - baseline_metrics['f1']
    
    interpretation['baseline_comparison'] = {
        'accuracy_improvement': float(acc_improvement),
        'f1_improvement': float(f1_improvement),
        'meaningful_improvement': bool(f1_improvement > 0.10),
        'assessment': (
            f"Model improves accuracy by {acc_improvement:.4f} and F1-score by {f1_improvement:.4f}. "
            f"{'This is meaningful improvement.' if f1_improvement > 0.10 else 'Improvement is marginal.'}"
        )
    }
    
    # 2. Class imbalance analysis
    total_samples = sum(class_distribution.values())
    majority_class = max(class_distribution, key=class_distribution.get)
    majority_pct = (class_distribution[majority_class] / total_samples) * 100
    is_imbalanced = majority_pct > 60.0
    
    interpretation['class_imbalance'] = {
        'is_imbalanced': bool(is_imbalanced),
        'majority_class': int(majority_class),
        'majority_percentage': float(majority_pct),
        'distribution': {int(k): int(v) for k, v in class_distribution.items()},
        'assessment': (
            f"Dataset is {'IMBALANCED' if is_imbalanced else 'BALANCED'} "
            f"with {majority_pct:.1f}% in class {majority_class}. "
            f"{'Accuracy alone is misleading.' if is_imbalanced else 'Accuracy is reliable.'}"
        )
    }
    
    # 3. Accuracy reliability
    interpretation['accuracy_reliability'] = {
        'is_reliable': bool(not is_imbalanced and model_metrics['accuracy'] > baseline_metrics['accuracy'] + 0.05),
        'reasoning': (
            "Accuracy is NOT reliable for this dataset because:\n"
            "  - Dataset is imbalanced (majority class dominates)\n"
            "  - A naive baseline achieves high accuracy by always predicting majority class\n"
            "  - Accuracy doesn't reflect minority class (readmission) detection performance"
            if is_imbalanced else
            "Accuracy is reliable for this dataset because:\n"
            "  - Dataset is balanced\n"
            "  - Model significantly outperforms baseline"
        )
    }
    
    # 4. Confusion matrix insights
    cm = np.array(model_metrics['confusion_matrix'])
    tn, fp, fn, tp = cm.ravel()
    
    minority_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    minority_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    interpretation['confusion_matrix_insights'] = {
        'true_positives': int(tp),
        'false_negatives': int(fn),
        'minority_class_recall': float(minority_recall),
        'minority_class_precision': float(minority_precision),
        'minority_class_detected': bool(tp > 0),
        'assessment': (
            f"Model detects {tp} out of {tp + fn} minority class samples (Recall: {minority_recall:.4f}). "
            f"{'Minority class is being detected.' if tp > 0 else 'Minority class is NOT being detected.'} "
            f"{'However, detection rate is low.' if minority_recall < 0.5 else 'Detection rate is acceptable.'}"
        )
    }
    
    # 5. Cross-validation stability
    cv_acc_std = cv_results['accuracy']['std']
    is_stable = cv_acc_std < 0.05
    
    interpretation['cross_validation'] = {
        'is_stable': bool(is_stable),
        'accuracy_std': float(cv_acc_std),
        'assessment': (
            f"Model is {'STABLE' if is_stable else 'UNSTABLE'} across folds "
            f"(std: {cv_acc_std:.4f}). "
            f"{'Performance is consistent.' if is_stable else 'Performance varies significantly.'}"
        )
    }
    
    # 6. Recommended metrics
    interpretation['recommended_metrics'] = {
        'primary': 'F1-score' if is_imbalanced else 'Accuracy',
        'secondary': ['Precision', 'Recall', 'Balanced Accuracy', 'ROC-AUC'],
        'reasoning': (
            "For imbalanced datasets, F1-score balances Precision and Recall, "
            "providing a better measure of model performance on minority class."
            if is_imbalanced else
            "For balanced datasets, Accuracy is sufficient, but Precision and Recall "
            "provide additional insights into model behavior."
        )
    }
    
    return interpretation


# ============================================================================
# Main Evaluation Pipeline
# ============================================================================

def main():
    """Execute the complete classification evaluation pipeline."""
    
    print_section_header("CLASSIFICATION MODEL EVALUATION USING ACCURACY")
    print("Assignment: Evaluating Classification Models Using Accuracy")
    print("Project: MEDILENS - Hospital Readmission Prediction")
    print(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # STEP 1: Load and Prepare Data
    # ========================================================================
    print_section_header("STEP 1: DATA LOADING AND PREPARATION", "-")
    
    print("Loading dataset...")
    df = load_data(DATA_PATH)
    validate_schema(df)
    df_clean = clean_data(df)
    
    print(f"✓ Dataset loaded: {df_clean.shape[0]:,} rows, {df_clean.shape[1]} columns")
    print(f"✓ Target column: {TARGET_COLUMN}")
    
    # Check class distribution
    class_dist = df_clean[TARGET_COLUMN].value_counts().to_dict()
    total = len(df_clean)
    print(f"\nClass Distribution:")
    for cls, count in sorted(class_dist.items()):
        print(f"  Class {cls}: {count:5d} ({count/total*100:5.2f}%)")
    
    # ========================================================================
    # STEP 2: Stratified Train-Test Split
    # ========================================================================
    print_section_header("STEP 2: STRATIFIED TRAIN-TEST SPLIT", "-")
    
    print(f"Splitting data with stratification...")
    print(f"  Test size: {TEST_SIZE}")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Stratify: {TARGET_COLUMN} (ensures balanced class distribution)")
    
    X_train, X_test, y_train, y_test = split_data(
        df_clean,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    
    print(f"\n✓ Train set: {len(X_train):,} samples")
    print(f"✓ Test set:  {len(X_test):,} samples")
    
    # Verify stratification
    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()
    
    print(f"\nTrain distribution:")
    for cls, pct in sorted(train_dist.items()):
        print(f"  Class {cls}: {pct*100:5.2f}%")
    
    print(f"\nTest distribution:")
    for cls, pct in sorted(test_dist.items()):
        print(f"  Class {cls}: {pct*100:5.2f}%")
    
    # ========================================================================
    # STEP 3: Feature Preprocessing
    # ========================================================================
    print_section_header("STEP 3: FEATURE PREPROCESSING", "-")
    
    print("Preprocessing features...")
    X_train_clean = drop_id_columns(X_train)
    X_test_clean = drop_id_columns(X_test)
    
    pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
    
    print("  Fitting preprocessing pipeline on training data only...")
    X_train_proc = pipeline.fit_transform(X_train_clean)
    
    print("  Transforming test data...")
    X_test_proc = pipeline.transform(X_test_clean)
    
    print(f"\n✓ Preprocessed shape: {X_train_proc.shape}")
    print(f"✓ Features: {X_train_proc.shape[1]}")
    
    # ========================================================================
    # STEP 4: Baseline Model (Majority Class)
    # ========================================================================
    print_section_header("STEP 4: BASELINE MODEL (MAJORITY CLASS)", "-")
    
    print("Training baseline model...")
    print("  Strategy: most_frequent (always predicts majority class)")
    
    baseline_model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline_model.fit(X_train_proc, y_train)
    
    print("✓ Baseline model trained")
    
    # Evaluate baseline on test set
    print("\nEvaluating baseline on test set...")
    baseline_pred = baseline_model.predict(X_test_proc)
    
    baseline_metrics = {
        'accuracy': accuracy_score(y_test, baseline_pred),
        'precision': precision_score(y_test, baseline_pred, zero_division=0),
        'recall': recall_score(y_test, baseline_pred, zero_division=0),
        'f1': f1_score(y_test, baseline_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, baseline_pred),
        'confusion_matrix': confusion_matrix(y_test, baseline_pred).tolist(),
    }
    
    print("\nBaseline Model Performance (Test Set):")
    print(f"  Accuracy:          {format_metric(baseline_metrics['accuracy'])}")
    print(f"  Precision:         {format_metric(baseline_metrics['precision'])}")
    print(f"  Recall:            {format_metric(baseline_metrics['recall'])}")
    print(f"  F1-Score:          {format_metric(baseline_metrics['f1'])}")
    print(f"  Balanced Accuracy: {format_metric(baseline_metrics['balanced_accuracy'])}")
    
    print_confusion_matrix_analysis(np.array(baseline_metrics['confusion_matrix']))
    
    # ========================================================================
    # STEP 5: Train Classification Model
    # ========================================================================
    print_section_header("STEP 5: TRAIN CLASSIFICATION MODEL", "-")
    
    print("Training RandomForestClassifier...")
    print(f"  Model parameters: {MODEL_PARAMS}")
    
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train_proc, y_train)
    
    print("✓ Model trained successfully")
    
    # ========================================================================
    # STEP 6: Evaluate Model on Test Set
    # ========================================================================
    print_section_header("STEP 6: MODEL EVALUATION ON TEST SET", "-")
    
    print("Evaluating model on held-out test set...")
    model_pred = model.predict(X_test_proc)
    
    model_metrics = {
        'accuracy': accuracy_score(y_test, model_pred),
        'precision': precision_score(y_test, model_pred, zero_division=0),
        'recall': recall_score(y_test, model_pred, zero_division=0),
        'f1': f1_score(y_test, model_pred, zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_test, model_pred),
        'confusion_matrix': confusion_matrix(y_test, model_pred).tolist(),
    }
    
    print("\nModel Performance (Test Set):")
    print(f"  Accuracy:          {format_metric(model_metrics['accuracy'])}")
    print(f"  Precision:         {format_metric(model_metrics['precision'])}")
    print(f"  Recall:            {format_metric(model_metrics['recall'])}")
    print(f"  F1-Score:          {format_metric(model_metrics['f1'])}")
    print(f"  Balanced Accuracy: {format_metric(model_metrics['balanced_accuracy'])}")
    
    print_confusion_matrix_analysis(np.array(model_metrics['confusion_matrix']))
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, model_pred, target_names=['Not Readmitted', 'Readmitted']))
    
    # ========================================================================
    # STEP 7: Cross-Validation
    # ========================================================================
    print_section_header("STEP 7: CROSS-VALIDATION ANALYSIS", "-")
    
    print("Performing 5-fold stratified cross-validation...")
    print("  This evaluates model stability across different data splits")
    
    cv_results = calculate_cross_validation_scores(
        model=RandomForestClassifier(**MODEL_PARAMS),
        X=X_train_proc,
        y=y_train,
        cv_folds=5,
        random_state=RANDOM_STATE
    )
    
    print("\nCross-Validation Results (Training Data):")
    print(f"  Accuracy:  {format_metric(cv_results['accuracy']['mean'])} ± {format_metric(cv_results['accuracy']['std'])}")
    print(f"  Precision: {format_metric(cv_results['precision']['mean'])} ± {format_metric(cv_results['precision']['std'])}")
    print(f"  Recall:    {format_metric(cv_results['recall']['mean'])} ± {format_metric(cv_results['recall']['std'])}")
    print(f"  F1-Score:  {format_metric(cv_results['f1']['mean'])} ± {format_metric(cv_results['f1']['std'])}")
    
    print(f"\nFold-by-fold Accuracy scores:")
    for i, score in enumerate(cv_results['accuracy']['scores'], 1):
        print(f"  Fold {i}: {format_metric(score)}")
    
    # ========================================================================
    # STEP 8: Model vs Baseline Comparison
    # ========================================================================
    print_section_header("STEP 8: MODEL VS BASELINE COMPARISON", "-")
    
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Balanced Accuracy'],
        'Baseline': [
            baseline_metrics['accuracy'],
            baseline_metrics['precision'],
            baseline_metrics['recall'],
            baseline_metrics['f1'],
            baseline_metrics['balanced_accuracy'],
        ],
        'Model': [
            model_metrics['accuracy'],
            model_metrics['precision'],
            model_metrics['recall'],
            model_metrics['f1'],
            model_metrics['balanced_accuracy'],
        ],
    })
    
    comparison_df['Improvement'] = comparison_df['Model'] - comparison_df['Baseline']
    comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Baseline']) * 100
    
    print("\nPerformance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # ========================================================================
    # STEP 9: Interpretation and Analysis
    # ========================================================================
    print_section_header("STEP 9: INTERPRETATION AND ANALYSIS", "-")
    
    interpretation = interpret_results(
        baseline_metrics=baseline_metrics,
        model_metrics=model_metrics,
        class_distribution=y_train.value_counts().to_dict(),
        cv_results=cv_results
    )
    
    print_subsection("Question 1: Does the model meaningfully outperform the baseline?")
    print(interpretation['baseline_comparison']['assessment'])
    print(f"\nKey Metrics:")
    print(f"  Accuracy improvement: {format_metric(interpretation['baseline_comparison']['accuracy_improvement'])}")
    print(f"  F1-score improvement: {format_metric(interpretation['baseline_comparison']['f1_improvement'])}")
    print(f"  Meaningful: {'YES' if interpretation['baseline_comparison']['meaningful_improvement'] else 'NO'}")
    
    print_subsection("Question 2: Is Accuracy a reliable metric for your dataset?")
    print(interpretation['accuracy_reliability']['reasoning'])
    print(f"\nReliable: {'YES' if interpretation['accuracy_reliability']['is_reliable'] else 'NO'}")
    
    print_subsection("Question 3: What does the confusion matrix reveal?")
    print(interpretation['confusion_matrix_insights']['assessment'])
    print(f"\nKey Findings:")
    print(f"  True Positives: {interpretation['confusion_matrix_insights']['true_positives']}")
    print(f"  False Negatives: {interpretation['confusion_matrix_insights']['false_negatives']}")
    print(f"  Minority Class Recall: {format_metric(interpretation['confusion_matrix_insights']['minority_class_recall'])}")
    print(f"  Minority Class Precision: {format_metric(interpretation['confusion_matrix_insights']['minority_class_precision'])}")
    
    print_subsection("Question 4: Are minority classes being detected properly?")
    if interpretation['confusion_matrix_insights']['minority_class_detected']:
        recall = interpretation['confusion_matrix_insights']['minority_class_recall']
        if recall >= 0.7:
            print(f"YES - Minority class detection is GOOD (Recall: {format_metric(recall)})")
        elif recall >= 0.5:
            print(f"PARTIALLY - Minority class detection is MODERATE (Recall: {format_metric(recall)})")
        else:
            print(f"NO - Minority class detection is POOR (Recall: {format_metric(recall)})")
    else:
        print("NO - Minority class is NOT being detected at all")
    
    print_subsection("Cross-Validation Stability")
    print(interpretation['cross_validation']['assessment'])
    
    print_subsection("Recommended Metrics for This Dataset")
    print(f"Primary metric: {interpretation['recommended_metrics']['primary']}")
    print(f"Secondary metrics: {', '.join(interpretation['recommended_metrics']['secondary'])}")
    print(f"\nReasoning: {interpretation['recommended_metrics']['reasoning']}")
    
    # ========================================================================
    # STEP 10: Save Results
    # ========================================================================
    print_section_header("STEP 10: SAVE EVALUATION RESULTS", "-")
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Save comprehensive report
    report = {
        'metadata': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset': DATA_PATH,
            'target_column': TARGET_COLUMN,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
        },
        'data_summary': {
            'total_samples': len(df_clean),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'class_distribution': class_dist,
            'train_distribution': y_train.value_counts().to_dict(),
            'test_distribution': y_test.value_counts().to_dict(),
        },
        'baseline_metrics': baseline_metrics,
        'model_metrics': model_metrics,
        'cross_validation': cv_results,
        'comparison': comparison_df.to_dict(orient='records'),
        'interpretation': interpretation,
    }
    
    report_path = os.path.join(REPORTS_DIR, "classification_accuracy_evaluation.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"✓ Comprehensive report saved: {report_path}")
    
    # Save comparison table as CSV
    csv_path = os.path.join(REPORTS_DIR, "baseline_vs_model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Comparison table saved: {csv_path}")
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print_section_header("EVALUATION COMPLETE")
    
    print("Key Takeaways:")
    print(f"  1. Model Accuracy: {format_metric(model_metrics['accuracy'])} vs Baseline: {format_metric(baseline_metrics['accuracy'])}")
    print(f"  2. Model F1-Score: {format_metric(model_metrics['f1'])} vs Baseline: {format_metric(baseline_metrics['f1'])}")
    print(f"  3. Dataset is {'IMBALANCED' if interpretation['class_imbalance']['is_imbalanced'] else 'BALANCED'}")
    print(f"  4. Accuracy is {'NOT ' if not interpretation['accuracy_reliability']['is_reliable'] else ''}reliable")
    print(f"  5. Recommended primary metric: {interpretation['recommended_metrics']['primary']}")
    
    print(f"\nReports saved to: {REPORTS_DIR}")
    print("\nNext steps:")
    print("  - Review the confusion matrix to understand prediction errors")
    print("  - Consider the scenario-based question for the video demo")
    print("  - Prepare video walkthrough of these results")
    
    print_section_header("", "=")


if __name__ == "__main__":
    main()
