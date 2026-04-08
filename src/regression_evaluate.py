"""
regression_evaluate.py
---------------------
Regression model evaluation metrics and functions.

Purpose
-------
Implement comprehensive evaluation for regression tasks following best practices:
- Compute MSE, RMSE, MAE, R², etc.
- Compare models against baseline (mean predictor)
- Provide interpretable metrics
- Support cross-validation

All metrics are defined from first principles in docstrings to make the
logic transparent and help practitioners understand what they're measuring.

Metrics Included
----------------
- MSE (Mean Squared Error): Training objective, penalizes large errors
- RMSE (Root MSE): Interpretable scale (same units as target)
- MAE (Mean Absolute Error): Robust to outliers
- R² (Coefficient of Determination): Variance explained
- MAPE (Mean Absolute Percentage Error): Scale-independent
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ---------------------------------------------------------------------------
# Core Regression Metrics
# ---------------------------------------------------------------------------

def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute comprehensive regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    dict
        Metrics dictionary with keys:
        - 'mse': Mean Squared Error
        - 'rmse': Root Mean Squared Error
        - 'mae': Mean Absolute Error
        - 'r2': R² Score (coefficient of determination)
        - 'mape': Mean Absolute Percentage Error

    Raises
    ------
    ValueError
        If y_true and y_pred have different lengths.

    Notes
    -----
    All metrics assume both inputs are 1D arrays or can be flattened to 1D.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} samples but "
            f"y_pred has {len(y_pred)} samples."
        )

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    metrics = {
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 4),
        "mape": round(float(mape), 4),
    }

    return metrics


def evaluate_regression_model(
    model,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """
    Evaluate a fitted regression model on held-out test data.

    Parameters
    ----------
    model : sklearn estimator
        Any fitted regressor that implements predict().
    X_test : np.ndarray
        Preprocessed feature matrix (must not have been seen during training).
    y_test : pd.Series or np.ndarray
        True target values aligned with X_test rows.

    Returns
    -------
    dict
        Regression metrics computed on test set.

    Raises
    ------
    ValueError
        If X_test and y_test have different numbers of rows.
    """
    if len(X_test) != len(y_test):
        raise ValueError(
            f"Shape mismatch: X_test has {len(X_test)} rows but "
            f"y_test has {len(y_test)} rows."
        )

    y_pred = model.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)

    return metrics


# ---------------------------------------------------------------------------
# Metric Definitions
# ---------------------------------------------------------------------------

def mean_absolute_percentage_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute Mean Absolute Percentage Error.

    MAPE = (1/n) * sum(|y_true - y_pred| / |y_true|) * 100

    Useful for:
    - Scale-independent metric (no units)
    - Comparing across different datasets
    - Understanding relative error magnitude

    Caveats:
    - Undefined when y_true contains zeros
    - Heavily penalizes underestimating when y_true is small
    - Not symmetric: error of ±10% is not the same for predictions below vs. above true value

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.

    Returns
    -------
    float
        MAPE as a percentage (0-100 typically, can exceed 100 if errors are large).
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0  # All true values are zero; MAPE is undefined

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0


# ---------------------------------------------------------------------------
# Metric Interpretation Helpers
# ---------------------------------------------------------------------------

def interpret_r2_score(r2: float) -> str:
    """
    Provide human-readable interpretation of R² score.

    Parameters
    ----------
    r2 : float
        R² score in range [-∞, 1.0].

    Returns
    -------
    str
        Interpretation of R² value.
    """
    if r2 >= 0.9:
        return "Excellent fit — explains >90% of variance"
    elif r2 >= 0.7:
        return "Strong fit — explains 70-90% of variance"
    elif r2 >= 0.5:
        return "Moderate fit — explains 50-70% of variance"
    elif r2 >= 0.3:
        return "Weak fit — explains 30-50% of variance"
    elif r2 >= 0.0:
        return "Poor fit — explains <30% of variance but beats baseline"
    elif r2 >= -0.1:
        return "Very poor fit — worse than baseline, but close"
    else:
        return "Model is actively worse than predicting the mean"


def metrics_comparison_summary(
    baseline_metrics: dict,
    model_metrics: dict,
) -> dict:
    """
    Compare baseline and model metrics with improvement calculations.

    Parameters
    ----------
    baseline_metrics : dict
        Metrics dict from evaluate_regression_model() on baseline model.
    model_metrics : dict
        Metrics dict from evaluate_regression_model() on primary model.

    Returns
    -------
    dict
        Comparison results with:
        - 'improvement_absolute': Absolute difference (model - baseline)
        - 'improvement_percentage': Relative improvement
        - 'better_on_metrics': Which model is better for each metric
        - 'summary': Text summary of comparison
    """
    improvement_abs = {}
    improvement_pct = {}
    better_model = {}

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        model_val = model_metrics[metric]

        improvement_abs[metric] = round(model_val - baseline_val, 4)

        # For metrics where lower is better (MSE, RMSE, MAE, MAPE),
        # positive improvement means model is better (lower error)
        # For R², higher is better, so positive improvement means model is better
        if metric == "r2":
            # For R², higher is better
            pct_change = ((model_val - baseline_val) /
                          max(abs(baseline_val), 0.001)) * 100
            better = "model" if model_val > baseline_val else "baseline"
        else:
            # For error metrics, lower is better
            pct_change = ((baseline_val - model_val) /
                          max(abs(baseline_val), 0.001)) * 100
            better = "model" if model_val < baseline_val else "baseline"

        improvement_pct[metric] = round(pct_change, 2)
        better_model[metric] = better

    summary = (
        f"Model improvement over baseline:\n"
        f"  RMSE: {improvement_pct['rmse']:+.1f}% "
        f"({'lower' if improvement_pct['rmse'] > 0 else 'higher'} is better)\n"
        f"  R²: {improvement_pct['r2']:+.1f}% "
        f"({'higher' if improvement_pct['r2'] > 0 else 'lower'} is better)"
    )

    return {
        "improvement_absolute": improvement_abs,
        "improvement_percentage": improvement_pct,
        "better_on_metrics": better_model,
        "summary": summary,
    }
