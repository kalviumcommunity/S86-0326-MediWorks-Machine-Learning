"""
model_comparison.py
-------------------
Compare baseline and primary model performance using identical metrics.

Purpose
-------
Provide side-by-side comparison of baseline and main model to:
- Demonstrate meaningful improvement over trivial solutions
- Identify which metrics show the most improvement
- Justify the complexity of the ML model
- Reveal dataset characteristics (imbalance, predictive signal)

Design Notes
------------
- Uses identical evaluation metrics for fair comparison
- Calculates absolute and relative improvement
- Highlights which model is better for each metric
- Generates comparison reports in multiple formats (dict, table, JSON)
"""

import json
import os
from typing import Dict, Any

import pandas as pd

from src.evaluate import evaluate_model


def compare_models(
    baseline_model,
    main_model,
    X_test,
    y_test,
) -> Dict[str, Any]:
    """
    Compare baseline and main model performance on test data.

    Both models are evaluated using identical metrics to ensure fair comparison.
    The function calculates absolute and relative improvement for each metric.

    Parameters
    ----------
    baseline_model : DummyClassifier
        Fitted baseline model.
    main_model : RandomForestClassifier
        Fitted primary ML model.
    X_test : np.ndarray
        Preprocessed test features.
    y_test : pd.Series
        True test labels.

    Returns
    -------
    dict
        Comparison results containing:
        - baseline_metrics: Dict of baseline performance
        - main_model_metrics: Dict of main model performance
        - improvement: Dict of absolute improvement (main - baseline)
        - improvement_percentage: Dict of relative improvement ((main - baseline) / baseline * 100)
        - better_model: Dict indicating which model is better for each metric
        - summary: Overall assessment

    Examples
    --------
    >>> comparison = compare_models(baseline, model, X_test, y_test)
    >>> print(comparison['summary']['overall_improvement'])
    'Main model shows significant improvement over baseline'
    """
    # Evaluate both models using identical metrics
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test)
    main_model_metrics = evaluate_model(main_model, X_test, y_test)

    # Calculate improvement
    improvement = {}
    improvement_pct = {}
    better_model = {}

    for metric in baseline_metrics.keys():
        baseline_val = baseline_metrics[metric]
        main_val = main_model_metrics[metric]

        # Absolute improvement
        abs_improvement = main_val - baseline_val
        improvement[metric] = round(abs_improvement, 4)

        # Relative improvement (percentage)
        if baseline_val > 0:
            rel_improvement = (abs_improvement / baseline_val) * 100
            improvement_pct[metric] = round(rel_improvement, 2)
        else:
            # Handle division by zero (baseline metric is 0)
            improvement_pct[metric] = float('inf') if main_val > 0 else 0.0

        # Which model is better
        better_model[metric] = "main_model" if main_val > baseline_val else "baseline"

    # Generate summary
    summary = _generate_comparison_summary(
        baseline_metrics,
        main_model_metrics,
        improvement,
        improvement_pct,
    )

    return {
        "baseline_metrics": baseline_metrics,
        "main_model_metrics": main_model_metrics,
        "improvement": improvement,
        "improvement_percentage": improvement_pct,
        "better_model": better_model,
        "summary": summary,
    }


def _generate_comparison_summary(
    baseline_metrics: dict,
    main_model_metrics: dict,
    improvement: dict,
    improvement_pct: dict,
) -> dict:
    """
    Generate a human-readable summary of the comparison.

    Parameters
    ----------
    baseline_metrics : dict
        Baseline model metrics.
    main_model_metrics : dict
        Main model metrics.
    improvement : dict
        Absolute improvement for each metric.
    improvement_pct : dict
        Relative improvement percentage for each metric.

    Returns
    -------
    dict
        Summary containing:
        - overall_improvement: Text assessment
        - key_improvements: List of metrics with significant improvement
        - concerns: List of metrics with little/no improvement
        - is_meaningful: Boolean indicating if improvement is meaningful
    """
    # Define thresholds for meaningful improvement
    MEANINGFUL_THRESHOLD = 0.10  # 10% absolute improvement
    SIGNIFICANT_THRESHOLD = 0.20  # 20% absolute improvement

    key_improvements = []
    concerns = []

    # Analyze each metric
    for metric, abs_imp in improvement.items():
        if abs_imp >= SIGNIFICANT_THRESHOLD:
            key_improvements.append(f"{metric}: +{abs_imp:.4f} ({improvement_pct[metric]:.1f}%)")
        elif abs_imp < MEANINGFUL_THRESHOLD:
            concerns.append(f"{metric}: +{abs_imp:.4f} (marginal improvement)")

    # Overall assessment
    avg_improvement = sum(improvement.values()) / len(improvement)
    is_meaningful = avg_improvement >= MEANINGFUL_THRESHOLD

    if avg_improvement >= SIGNIFICANT_THRESHOLD:
        overall = "Main model shows significant improvement over baseline"
    elif avg_improvement >= MEANINGFUL_THRESHOLD:
        overall = "Main model shows meaningful improvement over baseline"
    else:
        overall = "Main model shows marginal improvement over baseline"

    return {
        "overall_improvement": overall,
        "average_improvement": round(avg_improvement, 4),
        "key_improvements": key_improvements,
        "concerns": concerns if concerns else ["No major concerns"],
        "is_meaningful": is_meaningful,
    }


def print_comparison_table(comparison: dict) -> None:
    """
    Print a formatted comparison table to console.

    Parameters
    ----------
    comparison : dict
        Output from compare_models().
    """
    baseline = comparison["baseline_metrics"]
    main_model = comparison["main_model_metrics"]
    improvement = comparison["improvement"]
    improvement_pct = comparison["improvement_percentage"]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON: BASELINE vs MAIN MODEL")
    print("=" * 80)

    # Table header
    print(f"\n{'Metric':<15} {'Baseline':<12} {'Main Model':<12} {'Improvement':<15} {'% Change':<12}")
    print("-" * 80)

    # Table rows
    for metric in baseline.keys():
        base_val = baseline[metric]
        main_val = main_model[metric]
        imp_val = improvement[metric]
        imp_pct = improvement_pct[metric]

        # Format improvement with + sign
        imp_str = f"+{imp_val:.4f}" if imp_val >= 0 else f"{imp_val:.4f}"
        pct_str = f"+{imp_pct:.1f}%" if imp_pct != float('inf') else "+∞%"

        print(f"{metric:<15} {base_val:<12.4f} {main_val:<12.4f} {imp_str:<15} {pct_str:<12}")

    print("-" * 80)

    # Summary
    summary = comparison["summary"]
    print(f"\nSummary: {summary['overall_improvement']}")
    print(f"Average Improvement: {summary['average_improvement']:.4f}")

    if summary['key_improvements']:
        print("\nKey Improvements:")
        for imp in summary['key_improvements']:
            print(f"  ✓ {imp}")

    if summary['concerns'] and summary['concerns'] != ["No major concerns"]:
        print("\nConcerns:")
        for concern in summary['concerns']:
            print(f"  ⚠ {concern}")

    print("\n" + "=" * 80 + "\n")


def save_comparison_report(
    comparison: dict,
    baseline_description: dict,
    output_path: str,
) -> None:
    """
    Save comparison results to a JSON file.

    Parameters
    ----------
    comparison : dict
        Output from compare_models().
    baseline_description : dict
        Output from get_baseline_description().
    output_path : str
        Path to save the JSON report.
    """
    report = {
        "baseline_description": baseline_description,
        "comparison": comparison,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=4)


def generate_comparison_dataframe(comparison: dict) -> pd.DataFrame:
    """
    Generate a pandas DataFrame for easy visualization and export.

    Parameters
    ----------
    comparison : dict
        Output from compare_models().

    Returns
    -------
    pd.DataFrame
        Comparison table with metrics as rows and models as columns.
    """
    baseline = comparison["baseline_metrics"]
    main_model = comparison["main_model_metrics"]
    improvement = comparison["improvement"]

    df = pd.DataFrame({
        "Baseline": baseline,
        "Main Model": main_model,
        "Improvement": improvement,
    })

    return df
