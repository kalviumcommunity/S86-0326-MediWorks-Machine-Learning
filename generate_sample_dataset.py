"""Generate a synthetic hospital_visits.csv dataset.

Creates data/raw/hospital_visits.csv with the schema expected by the MEDILENS
pipeline. This is only for local testing/demo when a real dataset is not
available.

Usage:
    python generate_sample_dataset.py
    python generate_sample_dataset.py --rows 5000 --seed 7
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = np.array(["Emergency", "ICU", "General Medicine", "Surgery", "Pediatrics"])
    genders = np.array(["Male", "Female", "Other"])
    admission_types = np.array(["Emergency", "Elective", "Urgent"])
    bed_types = np.array(["General", "ICU", "Private"])

    patient_id = np.array([f"P{idx:07d}" for idx in range(1, rows + 1)])

    # Dates within last 2 years
    start = datetime.now() - timedelta(days=730)
    admission_offsets = rng.integers(0, 730, size=rows)
    admission_date = np.array([start + timedelta(days=int(d)) for d in admission_offsets])

    # Length of stay (days) with some long-stay tail
    los = rng.gamma(shape=2.0, scale=2.0, size=rows)  # mean ~4
    los = np.clip(los, 0.5, 30).round(2)
    discharge_date = np.array([admission_date[i] + timedelta(days=float(los[i])) for i in range(rows)])

    age = np.clip(rng.normal(55, 18, size=rows), 0, 95).round(0)
    num_procedures = np.clip(rng.poisson(2.0, size=rows), 0, 12)
    num_medications = np.clip(rng.poisson(7.0, size=rows), 0, 40)
    num_diagnoses = np.clip(rng.poisson(3.0, size=rows), 0, 15)

    department = rng.choice(departments, size=rows, p=[0.30, 0.12, 0.35, 0.15, 0.08])
    gender = rng.choice(genders, size=rows, p=[0.49, 0.49, 0.02])
    admission_type = rng.choice(admission_types, size=rows, p=[0.55, 0.25, 0.20])
    bed_type = rng.choice(bed_types, size=rows, p=[0.65, 0.20, 0.15])

    # Create readmission probability based on sensible risk factors
    # Higher risk: older, longer LOS, ICU/Emergency, more diagnoses/meds.
    dept_risk = np.where(department == "ICU", 0.7, 0.0) + np.where(department == "Emergency", 0.3, 0.0)
    admit_risk = np.where(admission_type == "Emergency", 0.5, 0.0)

    logits = (
        -3.0
        + 0.02 * (age - 50)
        + 0.10 * (los - 3)
        + 0.08 * (num_diagnoses - 3)
        + 0.03 * (num_medications - 7)
        + 0.05 * (num_procedures - 2)
        + dept_risk
        + admit_risk
    )
    p_readmit = _sigmoid(logits)
    readmitted = (rng.random(size=rows) < p_readmit).astype(int)

    df = pd.DataFrame(
        {
            "patient_id": patient_id,
            "admission_date": pd.to_datetime(admission_date),
            "discharge_date": pd.to_datetime(discharge_date),
            "department": department,
            "gender": gender,
            "admission_type": admission_type,
            "bed_type": bed_type,
            "age": age.astype(float),
            "length_of_stay": los.astype(float),
            "num_procedures": num_procedures.astype(float),
            "num_medications": num_medications.astype(float),
            "num_diagnoses": num_diagnoses.astype(float),
            "readmitted": readmitted,
        }
    )

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    out_path = os.path.join(raw_dir, "hospital_visits.csv")
    df = generate(rows=args.rows, seed=args.seed)
    df.to_csv(out_path, index=False)

    print(f"Wrote {len(df)} rows → {out_path}")
    print("Target distribution:")
    print(df["readmitted"].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    main()
