"""
eda.py
------
Inspect feature distributions before any model training begins.

Responsibilities
----------------
- Load raw dataset for exploratory analysis
- Summarize numerical feature distributions and skewness
- Generate histograms and boxplots for numerical features
- Inspect categorical distributions and identify rare labels
- Compare selected numerical features across target classes

This module does not perform model training or fit preprocessing pipelines.
It is intended for safe, reproducible inspection of the dataset before any
modeling decisions are made.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import (
    DATA_PATH,
    REPORTS_DIR,
    RAW_DATA_DIR,
    RANDOM_STATE,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    TARGET_COLUMN,
)


def ensure_report_dir():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset from disk.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the provided filepath.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw dataset not found at '{filepath}'. "
            "Place the dataset in data/raw/ before running EDA."
        )

    df = pd.read_csv(filepath, parse_dates=["admission_date", "discharge_date"], infer_datetime_format=True)
    return df


def generate_sample_dataset(filepath: str = DATA_PATH, n_rows: int = 1000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a synthetic dataset for EDA when raw data is unavailable.

    This helper is intended for demonstration only. It creates a reproducible
    synthetic dataset with the same schema expected by the pipeline.
    """
    rng = np.random.default_rng(random_state)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    departments = ["Emergency", "ICU", "General Medicine", "Surgery", "Pediatrics"]
    genders = ["Male", "Female", "Other"]
    admission_types = ["Emergency", "Elective", "Urgent"]
    bed_types = ["General", "ICU", "Private"]

    admission_dates = pd.date_range(start="2024-01-01", periods=n_rows, freq="H")
    discharge_dates = admission_dates + pd.to_timedelta(rng.integers(1, 15, size=n_rows), unit="D")

    df = pd.DataFrame({
        "patient_id": [f"P{100000 + i}" for i in range(n_rows)],
        "admission_date": admission_dates,
        "discharge_date": discharge_dates,
        "department": rng.choice(departments, size=n_rows, p=[0.30, 0.15, 0.30, 0.15, 0.10]),
        "gender": rng.choice(genders, size=n_rows, p=[0.48, 0.50, 0.02]),
        "admission_type": rng.choice(admission_types, size=n_rows, p=[0.60, 0.25, 0.15]),
        "bed_type": rng.choice(bed_types, size=n_rows, p=[0.65, 0.20, 0.15]),
        "age": np.clip(rng.normal(55, 18, size=n_rows), 0, 95).round(1),
        "length_of_stay": np.clip(rng.normal(5, 3, size=n_rows), 1, 30).round(1),
        "num_procedures": np.clip(rng.poisson(2, size=n_rows), 0, 10),
        "num_medications": np.clip(rng.poisson(5, size=n_rows), 0, 20),
        "num_diagnoses": np.clip(rng.poisson(3, size=n_rows), 0, 10),
    })

    # Introduce a binary readmission target with a mild signal
    base_prob = 0.15 + (df["length_of_stay"] > 7) * 0.10 + (df["age"] > 65) * 0.05
    target = rng.random(n_rows) < base_prob
    df[TARGET_COLUMN] = target.astype(int)

    df.to_csv(filepath, index=False)
    return df


def describe_numeric_features(df: pd.DataFrame, numerical_cols: list[str] = NUMERICAL_COLS) -> pd.DataFrame:
    """Compute summary statistics and skewness for numerical features."""
    summary = df[numerical_cols].describe().T
    summary["skewness"] = df[numerical_cols].skew().round(4)
    return summary


def plot_numeric_distributions(df: pd.DataFrame, numerical_cols: list[str] = NUMERICAL_COLS) -> dict:
    """Generate histograms and boxplots for numerical features.

    Returns a dict mapping plot descriptions to saved file paths.
    """
    ensure_report_dir()
    plots = {}

    # Histograms
    fig, axs = plt.subplots(len(numerical_cols), 1, figsize=(8, 4 * len(numerical_cols)))
    if len(numerical_cols) == 1:
        axs = [axs]

    for ax, col in zip(axs, numerical_cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram: {col}")

    histogram_path = os.path.join(REPORTS_DIR, "numerical_feature_histograms.png")
    fig.tight_layout()
    fig.savefig(histogram_path, dpi=150)
    plt.close(fig)
    plots["histograms"] = histogram_path

    # Boxplots
    fig, axs = plt.subplots(len(numerical_cols), 1, figsize=(8, 4 * len(numerical_cols)))
    if len(numerical_cols) == 1:
        axs = [axs]

    for ax, col in zip(axs, numerical_cols):
        sns.boxplot(x=df[col], ax=ax)
        ax.set_title(f"Boxplot: {col}")

    boxplot_path = os.path.join(REPORTS_DIR, "numerical_feature_boxplots.png")
    fig.tight_layout()
    fig.savefig(boxplot_path, dpi=150)
    plt.close(fig)
    plots["boxplots"] = boxplot_path

    return plots


def identify_numeric_outliers(df: pd.DataFrame, numerical_cols: list[str] = NUMERICAL_COLS) -> pd.DataFrame:
    """Identify potential outliers using the IQR rule."""
    records = []
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        records.append(
            {
                "feature": col,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(outliers.count()),
                "outlier_pct": round(outliers.count() / len(df) * 100, 2),
                "min_value": float(df[col].min()),
                "max_value": float(df[col].max()),
            }
        )
    return pd.DataFrame(records)


def inspect_categorical_features(df: pd.DataFrame, categorical_cols: list[str] = CATEGORICAL_COLS) -> dict:
    """Inspect categorical distribution, rare categories, and inconsistent labels."""
    results = {}
    for col in categorical_cols:
        counts = df[col].astype(str).value_counts(dropna=False)
        normalized = df[col].astype(str).value_counts(normalize=True, dropna=False).round(4)
        rare = counts[counts < max(1, len(df) * 0.01)]
        normalized_rare = normalized.loc[rare.index]
        results[col] = {
            "counts": counts,
            "frequencies": normalized,
            "rare_categories": rare,
            "rare_percentage": normalized_rare.sum(),
            "unique_values": int(counts.shape[0]),
        }
    return results


def compare_numeric_by_target(df: pd.DataFrame, features: list[str], target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """Compare numerical feature summaries across target classes."""
    grouped = df.groupby(target_column)[features].describe().stack(level=0)
    return grouped


def plot_compare_by_target(df: pd.DataFrame, features: list[str], target_column: str = TARGET_COLUMN) -> str:
    """Generate grouped boxplots for selected numerical features by target class."""
    ensure_report_dir()
    fig, axs = plt.subplots(len(features), 1, figsize=(8, 4 * len(features)))
    if len(features) == 1:
        axs = [axs]

    for ax, feature in zip(axs, features):
        sns.boxplot(x=target_column, y=feature, data=df, ax=ax)
        ax.set_title(f"{feature} distribution by {target_column}")
        ax.set_xlabel(target_column)

    path = os.path.join(REPORTS_DIR, "numeric_comparison_by_target.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
