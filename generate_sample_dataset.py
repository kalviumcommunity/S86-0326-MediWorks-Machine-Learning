"""generate_sample_dataset.py
---------------------------
Create a small synthetic dataset that matches MEDILENS' expected schema.

This script exists to make the project runnable end-to-end even when you do not
have access to a real hospital dataset.

It writes a CSV to `data/raw/hospital_visits.csv` by default.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import (
    DATA_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    ID_COLUMNS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    DATETIME_COLS,
)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_hospital_visits(
    rows: int = 1000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = np.array(
        ["Emergency", "ICU", "General Medicine", "Surgery", "Pediatrics"])
    genders = np.array(["Male", "Female", "Other"])
    admission_types = np.array(["Emergency", "Elective", "Urgent"])
    bed_types = np.array(["General", "ICU", "Private"])

    # Dates: spread admissions across the last ~365 days
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)
    admission_offsets = rng.integers(0, 365, size=rows)
    admission_date = pd.to_datetime(
        [start_date + timedelta(days=int(d)) for d in admission_offsets])

    # Core numeric features
    age = np.clip(rng.normal(loc=52, scale=18, size=rows),
                  0, 95).round(0).astype(int)
    num_procedures = np.clip(rng.poisson(
        lam=2.0, size=rows), 0, 12).astype(int)
    num_medications = np.clip(rng.poisson(
        lam=8.0, size=rows), 0, 40).astype(int)
    num_diagnoses = np.clip(rng.poisson(lam=4.0, size=rows), 0, 20).astype(int)

    # Length of stay in days (right-skewed)
    length_of_stay = np.clip(
        rng.gamma(shape=2.0, scale=2.0, size=rows), 0.5, 30.0).round(2)
    discharge_date = admission_date + pd.to_timedelta(length_of_stay, unit="D")

    department = rng.choice(departments, size=rows, replace=True)
    gender = rng.choice(genders, size=rows, replace=True)
    admission_type = rng.choice(admission_types, size=rows, replace=True)
    bed_type = rng.choice(bed_types, size=rows, replace=True)

    # Build a simple synthetic readmission probability
    emergency = (department == "Emergency").astype(float)
    icu = (department == "ICU").astype(float)
    urgent_admission = (admission_type != "Elective").astype(float)

    logit = (
        -2.2
        + 0.020 * (age - 50)
        + 0.060 * (length_of_stay - 4)
        + 0.090 * (num_diagnoses - 4)
        + 0.030 * (num_medications - 8)
        + 0.50 * emergency
        + 0.85 * icu
        + 0.25 * urgent_admission
        + rng.normal(0, 0.35, size=rows)
    )
    p = _sigmoid(logit)
    readmitted = (rng.random(rows) < p).astype(int)

    patient_id = [f"P{idx:06d}" for idx in range(1, rows + 1)]

    df = pd.DataFrame(
        {
            ID_COLUMNS[0]: patient_id,
            DATETIME_COLS[0]: admission_date,
            DATETIME_COLS[1]: discharge_date,
            "department": department,
            "gender": gender,
            "admission_type": admission_type,
            "bed_type": bed_type,
            "age": age,
            "length_of_stay": length_of_stay,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "num_diagnoses": num_diagnoses,
            TARGET_COLUMN: readmitted,
        }
    )

    # Introduce a small amount of missingness to exercise imputers
    for col in ["department", "gender", "admission_type", "bed_type"]:
        mask = rng.random(rows) < 0.02
        df.loc[mask, col] = None

    for col in ["age", "num_procedures", "num_medications", "num_diagnoses", "length_of_stay"]:
        mask = rng.random(rows) < 0.02
        df.loc[mask, col] = np.nan

    # Ensure column order roughly matches config grouping
    ordered_cols = (
        ID_COLUMNS
        + DATETIME_COLS
        + CATEGORICAL_COLS
        + NUMERICAL_COLS
        + [TARGET_COLUMN]
    )
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic MEDILENS dataset")
    parser.add_argument("--rows", type=int, default=1000,
                        help="Number of rows to generate")
    parser.add_argument("--seed", type=int,
                        default=RANDOM_STATE, help="Random seed")
    parser.add_argument(
        "--out",
        type=str,
        default=DATA_PATH,
        help="Output CSV path (default: data/raw/hospital_visits.csv)",
    )
    args = parser.parse_args()

    df = generate_synthetic_hospital_visits(rows=args.rows, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} rows → {args.out}")


if __name__ == "__main__":
    main()
