"""
generate_sample_dataset.py
--------------------------
Create a synthetic hospital readmission dataset at data/raw/hospital_visits.csv.

Run
---
    python generate_sample_dataset.py
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.config import DATA_PATH, RAW_DATA_DIR, RANDOM_STATE


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n_rows: int = 1000, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate a realistic synthetic dataset for the MEDILENS pipeline."""
    rng = np.random.default_rng(random_state)

    departments = np.array(["Emergency", "ICU", "General Medicine", "Surgery", "Pediatrics"])
    genders = np.array(["Male", "Female", "Other"])
    admission_types = np.array(["Emergency", "Elective", "Urgent"])
    bed_types = np.array(["General", "ICU", "Private"])

    patient_ids = [f"PID{100000 + i}" for i in range(n_rows)]

    start_date = datetime(2024, 1, 1)
    admission_offsets = rng.integers(0, 365, size=n_rows)
    admission_dates = np.array([start_date + timedelta(days=int(d)) for d in admission_offsets])

    ages = rng.integers(18, 91, size=n_rows)
    length_of_stay = np.clip(rng.gamma(shape=2.5, scale=2.0, size=n_rows), 1.0, 30.0)
    discharge_dates = np.array(
        [admission_dates[i] + timedelta(days=float(length_of_stay[i])) for i in range(n_rows)]
    )

    num_procedures = rng.poisson(lam=2.5, size=n_rows)
    num_medications = rng.poisson(lam=8.0, size=n_rows)
    num_diagnoses = rng.poisson(lam=3.0, size=n_rows)

    department = rng.choice(departments, size=n_rows, p=[0.32, 0.12, 0.30, 0.16, 0.10])
    gender = rng.choice(genders, size=n_rows, p=[0.49, 0.49, 0.02])
    admission_type = rng.choice(admission_types, size=n_rows, p=[0.45, 0.25, 0.30])
    bed_type = rng.choice(bed_types, size=n_rows, p=[0.72, 0.16, 0.12])

    # Synthetic readmission probability with plausible clinical drivers.
    risk_linear = (
        -2.2
        + 0.018 * (ages - 50)
        + 0.11 * (length_of_stay - 5)
        + 0.22 * (num_diagnoses - 3)
        + 0.09 * (num_procedures - 2)
        + 0.05 * (num_medications - 8)
        + 0.7 * (department == "ICU").astype(float)
        + 0.45 * (department == "Emergency").astype(float)
        + 0.35 * (admission_type == "Emergency").astype(float)
    )
    readmission_prob = _sigmoid(risk_linear)
    readmitted = rng.binomial(1, np.clip(readmission_prob, 0.03, 0.97), size=n_rows)

    df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "admission_date": pd.to_datetime(admission_dates),
            "discharge_date": pd.to_datetime(discharge_dates),
            "department": department,
            "gender": gender,
            "admission_type": admission_type,
            "bed_type": bed_type,
            "age": ages,
            "length_of_stay": np.round(length_of_stay, 2),
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "num_diagnoses": num_diagnoses,
            "readmitted": readmitted,
        }
    )

    return df


def main() -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)

    positive_rate = float(df["readmitted"].mean())
    print(f"Saved synthetic dataset to: {DATA_PATH}")
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]} | Readmission rate: {positive_rate:.3f}")


if __name__ == "__main__":
    main()
