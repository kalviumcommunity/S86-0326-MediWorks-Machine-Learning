"""
problem_definition.py
---------------------
Utilities to identify supervised learning problem type from the target column.

This module provides a lightweight, data-driven check that should run before
model training so algorithm and metric choices are explicit and auditable.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_binary_integer_like(values: pd.Series) -> bool:
    """Return True if all non-null values are integer-like and >= 0."""
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return False
    return bool(((vals >= 0) & np.isclose(vals, np.round(vals))).all())


def infer_supervised_problem_type(
    y: pd.Series,
    imbalance_ratio_threshold: float = 0.20,
) -> dict[str, Any]:
    """
    Infer supervised problem type and subtype from a target series.

    Parameters
    ----------
    y : pd.Series
        Target variable.
    imbalance_ratio_threshold : float
        Minimum minority/majority ratio to consider classes balanced.

    Returns
    -------
    dict[str, Any]
        Structured summary including:
        - task_type: classification or regression
        - subtype: binary, multi-class, multi-label, continuous, or count
        - class_distribution and imbalance flags when applicable
        - suggested metrics and model families
    """
    if y is None or len(y) == 0:
        raise ValueError("Target series is empty; cannot infer problem type.")

    y_non_null = y.dropna()
    if y_non_null.empty:
        raise ValueError("Target series has only null values; cannot infer problem type.")

    n_samples = int(len(y_non_null))
    n_unique = int(y_non_null.nunique())

    # Heuristic 1: object/category/bool is classification by default.
    if pd.api.types.is_object_dtype(y_non_null) or pd.api.types.is_categorical_dtype(y_non_null) or pd.api.types.is_bool_dtype(y_non_null):
        task_type = "classification"
    else:
        # Heuristic 2: numeric with small unique set is likely classification.
        # Otherwise treat as regression.
        unique_ratio = n_unique / n_samples
        task_type = "classification" if n_unique <= 20 and unique_ratio <= 0.05 else "regression"

    result: dict[str, Any] = {
        "n_samples": n_samples,
        "n_unique": n_unique,
        "task_type": task_type,
    }

    if task_type == "classification":
        class_counts = y_non_null.value_counts(dropna=False)
        class_distribution = {str(k): int(v) for k, v in class_counts.items()}

        if n_unique == 2:
            subtype = "binary"
        else:
            subtype = "multi-class"

        majority = int(class_counts.max())
        minority = int(class_counts.min())
        minority_majority_ratio = round(minority / majority, 4) if majority else 0.0
        is_imbalanced = minority_majority_ratio < imbalance_ratio_threshold

        result.update(
            {
                "subtype": subtype,
                "class_distribution": class_distribution,
                "minority_majority_ratio": minority_majority_ratio,
                "is_imbalanced": is_imbalanced,
                "suggested_metrics": [
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ],
                "suggested_models": [
                    "LogisticRegression",
                    "RandomForestClassifier",
                    "GradientBoostingClassifier",
                    "XGBoost/LightGBM",
                ],
                "notes": [
                    "If classes are imbalanced, avoid relying on accuracy alone.",
                    "Tune decision threshold based on false-positive vs false-negative cost.",
                ],
            }
        )
    else:
        is_count = _is_binary_integer_like(y_non_null)
        subtype = "count" if is_count else "continuous"

        result.update(
            {
                "subtype": subtype,
                "target_summary": {
                    "min": float(pd.to_numeric(y_non_null, errors="coerce").min()),
                    "max": float(pd.to_numeric(y_non_null, errors="coerce").max()),
                    "mean": float(pd.to_numeric(y_non_null, errors="coerce").mean()),
                },
                "suggested_metrics": ["mae", "rmse", "r2"],
                "suggested_models": [
                    "LinearRegression",
                    "Ridge/Lasso",
                    "RandomForestRegressor",
                    "GradientBoostingRegressor",
                ],
                "notes": [
                    "Use count-specific models (Poisson/Negative Binomial) when target is count-like.",
                    "Inspect residual plots for systematic errors.",
                ],
            }
        )

    return result
