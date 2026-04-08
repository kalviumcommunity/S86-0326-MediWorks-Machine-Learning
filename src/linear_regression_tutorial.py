"""
linear_regression_tutorial.py
-----------------------------
Complete Linear Regression workflow example: from baseline comparison to
coefficient interpretation and residual analysis.

This script implements the standard pattern recommended in the lesson:

1. Load and prepare data
2. Split (train/test)
3. Train baseline (mean predictor)
4. Train Linear Regression model
5. Evaluate and compare
6. Inspect coefficients
7. Analyze residuals
8. Check cross-validation stability

Run this to understand the complete Linear Regression workflow.

Usage
-----
    python -m src.linear_regression_tutorial

Output
------
Prints metrics, comparison, coefficients, and residual statistics.
Demonstrates when and why Linear Regression works.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.linear_regression import (
    train_linear_regression,
    train_linear_regression_with_scaling,
    train_ridge_regression,
    train_lasso_regression,
    train_baseline_regressor,
    get_coefficients_dataframe,
    compute_residuals,
)
from src.regression_evaluate import (
    compute_regression_metrics,
    evaluate_regression_model,
    metrics_comparison_summary,
    interpret_r2_score,
)
from src.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Example Workflow
# ---------------------------------------------------------------------------

def example_complete_workflow(X, y, feature_names):
    """
    Complete Linear Regression workflow end-to-end.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Feature matrix.
    y : np.ndarray or pd.Series
        Target vector (continuous).
    feature_names : list
        Names of features for coefficient interpretation.
    """
    print("\n" + "="*70)
    print("COMPLETE LINEAR REGRESSION WORKFLOW")
    print("="*70)

    # Step 1: Split
    print("\n[Step 1: Train/Test Split]")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")

    # Step 2: Baseline
    print("\n[Step 2: Train Baseline (Mean Predictor)]")
    baseline = train_baseline_regressor(X_train, y_train, strategy="mean")
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = compute_regression_metrics(y_test, baseline_pred)
    print(f"  Baseline (mean) metrics on test set:")
    for metric, value in baseline_metrics.items():
        print(f"    {metric.upper():5s}: {value:.4f}")

    # Step 3: Linear Regression (unscaled)
    print("\n[Step 3: Train Linear Regression (Unscaled)]")
    model_unscaled = train_linear_regression(X_train, y_train)
    pred_unscaled = model_unscaled.predict(X_test)
    metrics_unscaled = compute_regression_metrics(y_test, pred_unscaled)
    print(f"  Linear Regression metrics on test set:")
    for metric, value in metrics_unscaled.items():
        print(f"    {metric.upper():5s}: {value:.4f}")

    # Step 4: Linear Regression with Scaling
    print("\n[Step 4: Train Linear Regression (Scaled)]")
    pipeline = train_linear_regression_with_scaling(X_train, y_train)
    pred_scaled = pipeline.predict(X_test)
    metrics_scaled = compute_regression_metrics(y_test, pred_scaled)
    print(f"  Linear Regression (with scaling) metrics on test set:")
    for metric, value in metrics_scaled.items():
        print(f"    {metric.upper():5s}: {value:.4f}")

    # Step 5: Comparison
    print("\n[Step 5: Model vs Baseline Comparison]")
    comparison = metrics_comparison_summary(baseline_metrics, metrics_scaled)
    print(f"\n  Absolute Improvement (model - baseline):")
    for metric, value in comparison["improvement_absolute"].items():
        better = "✓" if comparison["better_on_metrics"][metric] == "model" else "✗"
        print(f"    {metric.upper():5s}: {value:+.4f} {better}")

    print(f"\n  Relative Improvement (%):")
    for metric, value in comparison["improvement_percentage"].items():
        print(f"    {metric.upper():5s}: {value:+.2f}%")

    print(f"\n  {comparison['summary']}")

    # Step 6: Coefficient Interpretation
    print("\n[Step 6: Coefficient Interpretation]")
    lr_model = pipeline.named_steps["model"]
    coef_df = get_coefficients_dataframe(lr_model, feature_names)
    print(f"\n  Intercept: {lr_model.intercept_:.4f}")
    print(f"\n  Feature Coefficients (sorted by magnitude):")
    print(coef_df.to_string(index=False))
    print(f"\n  Interpretation notes:")
    print(f"    - Coefficients are standardized (features were scaled)")
    print(f"    - Magnitude indicates feature importance")
    print(f"    - Sign indicates direction of effect")
    print(f"    - Directly comparable across features (same scale)")

    # Step 7: Residual Analysis
    print("\n[Step 7: Residual Analysis]")
    residuals = compute_residuals(y_test, pred_scaled)
    print(f"  Residuals statistics:")
    print(f"    Mean: {np.mean(residuals):.6f} (should be ≈ 0)")
    print(f"    Std:  {np.std(residuals):.4f}")
    print(f"    Min:  {np.min(residuals):.4f}")
    print(f"    Max:  {np.max(residuals):.4f}")
    print(f"  ✓ If mean ≈ 0, model is unbiased")
    print(f"  ✓ If std is constant across predictions, assumption of")
    print(f"    homoscedasticity is reasonable")

    # Step 8: Cross-Validation
    print("\n[Step 8: Cross-Validation (5-fold)]")
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )
    print(f"  CV R² scores: {cv_scores}")
    print(f"  Mean CV R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  ✓ Low std across folds suggests stable model")
    print(f"  ✗ High std may indicate instability or multicollinearity")

    # Step 9: Model Selection (Regularization Comparison)
    print("\n[Step 9: Regularization Variants Comparison]")
    ridge_model = train_ridge_regression(X_train, y_train, alpha=10.0)
    ridge_pred = ridge_model.predict(X_test)
    ridge_metrics = compute_regression_metrics(y_test, ridge_pred)

    lasso_model = train_lasso_regression(X_train, y_train, alpha=0.1)
    lasso_pred = lasso_model.predict(X_test)
    lasso_metrics = compute_regression_metrics(y_test, lasso_pred)

    variant_comparison = pd.DataFrame({
        "Linear Regression": metrics_scaled,
        "Ridge (α=10)": ridge_metrics,
        "Lasso (α=0.1)": lasso_metrics,
    })
    print("\n  Model Variant Comparison:")
    print(variant_comparison.to_string())

    # Step 10: R² Interpretation
    print(f"\n[Step 10: R² Interpretation]")
    r2_value = metrics_scaled["r2"]
    interpretation = interpret_r2_score(r2_value)
    print(f"  R² = {r2_value:.4f}")
    print(f"  Interpretation: {interpretation}")

    # Practical Checklist
    print("\n[Practical Checklist Before Declaring Success]")
    checklist_items = [
        ("Model beats baseline on RMSE, MAE, R²",
         metrics_scaled["r2"] > baseline_metrics["r2"]),
        ("R² is meaningfully positive", metrics_scaled["r2"] > 0.3),
        ("Cross-validation R² is consistent", cv_scores.std() < 0.1),
        ("Residuals mean is close to 0", abs(
            np.mean(residuals)) < 0.1 * np.std(residuals)),
    ]
    for item, status in checklist_items:
        symbol = "✓" if status else "✗"
        print(f"  {symbol} {item}")

    print("\n" + "="*70 + "\n")


def generate_synthetic_regression_data(n_samples=200, n_features=5, random_state=42):
    """
    Generate synthetic regression data with known linear relationship.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    feature_names : list
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    # True linear relationship
    true_coef = np.array([2.5, -1.3, 0.8, 3.2, -0.5])
    y = X @ true_coef + np.random.randn(n_samples) * 0.5

    feature_names = [f"Feature_{i+1}" for i in range(n_features)]
    return X, y, feature_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LINEAR REGRESSION LEARNING WORKFLOW")
    print("Comprehensive tutorial on training, evaluating, and interpreting")
    print("Linear Regression models with best practices.")
    print("="*70)

    # Generate example data
    print("\n[Generating synthetic regression data]")
    X, y, feature_names = generate_synthetic_regression_data(
        n_samples=200,
        n_features=5,
        random_state=RANDOM_STATE
    )
    print(f"  Data shape: {X.shape}")
    print(f"  Features: {feature_names}")

    # Run complete workflow
    example_complete_workflow(X, y, feature_names)

    print("Tutorial complete!")
    print("\nKey Takeaways:")
    print("  1. Always compare against a baseline")
    print("  2. Use separate train/test splits (no leakage)")
    print("  3. Scale features for coefficient interpretation and stability")
    print("  4. Use pipelines to prevent scaling leakage")
    print("  5. Check cross-validation for stability")
    print("  6. Analyze residuals to validate assumptions")
    print("  7. Regularization (Ridge/Lasso) helps with multicollinearity")
    print("  8. R² alone doesn't tell the full story—always contextualize")
