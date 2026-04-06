# MEDILENS — AI-Powered Hospital Visit Analytics & Resource Optimization System

**Project Name:** MEDILENS
**Team Name:** MediWorks

MEDILENS is an AI and Machine Learning-based hospital analytics platform that analyses patient admission and discharge data to identify peak admission times, calculate average length of stay (LOS), detect overloaded departments, and **predict patient readmission risk** for better staffing and resource planning.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Proposed Solution](#proposed-solution)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Tech Stack](#tech-stack)
- [Dataset Format](#dataset-format)
- [Project Structure](#project-structure)
- [ML Pipeline Architecture](#ml-pipeline-architecture)
- [Data Splitting Strategy](#data-splitting-strategy)
- [Installation & Setup](#installation--setup)
- [Running the Pipeline](#running-the-pipeline)
- [API Endpoints (Sample)](#api-endpoints-sample)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

Hospitals generate large volumes of patient visit data every day — admissions, discharges, department transfers, and bed occupancy. Most hospitals still depend on manual reporting and administrator intuition for staffing and resource decisions.

MEDILENS solves this using AI-driven analytics and Machine Learning forecasting to support data-driven hospital resource planning, including a **30-day readmission prediction model** built on a modular Python ML pipeline.

---

## Problem Statement

Hospitals rely on intuition instead of data-driven insights, leading to:

* Staff shortages during peak hours
* Department overcrowding
* Poor bed management
* Delayed discharges
* Increased waiting time

---

## Proposed Solution

MEDILENS provides:

* Peak admission analysis
* LOS (Length of Stay) calculation
* Department overload detection
* Readmission prediction
* Bed occupancy forecasting
* Staffing recommendations

---

## Key Features

### Hospital Analytics

* Admission trend analysis
* Peak time detection
* Department-wise insights

### LOS Analysis

* Overall LOS
* Department-wise LOS

### ML Predictions

* Readmission prediction
* Admission forecasting

### Modular Pipeline

* Clean architecture
* Independent modules

---

## System Workflow

```
Raw CSV → load_data() → validate_schema() → clean_data()
        → split_data()
        → preprocessing.fit(X_train)
        → transform(X_test)
        → train_model()
        → evaluate_model()
        → save_artifacts()
```

---

## Tech Stack

### Machine Learning

* Python
* Pandas, NumPy
* Scikit-learn
* Joblib

### Frontend (Planned)

* React.js
* Chart.js

### Backend (Planned)

* FastAPI

---

## Dataset Format

The pipeline expects `data/raw/hospital_visits.csv` with the following columns:

| Column Name       | Type        | Description                              |
|------------------|-------------|------------------------------------------|
| `patient_id`      | string      | Unique patient identifier (dropped before model) |
| `admission_date`  | datetime    | Admission date                           |
| `discharge_date`  | datetime    | Discharge date                           |
| `department`      | categorical | Emergency / ICU / General Medicine / Surgery / Pediatrics |
| `gender`          | categorical | Male / Female / Other                    |
| `admission_type`  | categorical | Emergency / Elective / Urgent            |
| `bed_type`        | categorical | General / ICU / Private                  |
| `age`             | float       | Patient age in years                     |
| `length_of_stay`  | float       | Days between admission and discharge     |
| `num_procedures`  | float       | Number of procedures during visit        |
| `num_medications` | float       | Number of medications administered       |
| `num_diagnoses`   | float       | Number of recorded diagnoses             |
| `readmitted`      | int (0/1)   | Target: 1 = readmitted within 30 days   |

---

## Feature and Target Definitions

All definitions are centralized in `src/config.py` so every pipeline stage uses the exact same target and feature list.

### Target Variable (y)
- **Column:** `readmitted`
- **Type:** Binary classification
- **Values (business meaning):**
      - `1` = patient was readmitted within 30 days
      - `0` = patient was not readmitted within 30 days
- **What a correct prediction means:** accurately estimating readmission risk so hospitals can plan staffing, bed allocation, and follow-up interventions.

### Feature Columns (X)
All features are chosen using the rule: **“Would I have this information at prediction time?”**

**Numerical Features**
- `age` — patient age
- `length_of_stay` — days stayed (computed from admission/discharge if needed)
- `num_procedures` — number of procedures during the visit
- `num_medications` — number of medications administered
- `num_diagnoses` — number of diagnoses recorded

**Categorical Features**
- `department` — department of care (e.g., ICU, Emergency)
- `gender` — patient gender
- `admission_type` — emergency/elective/urgent
- `bed_type` — bed category (General/ICU/Private)

### Excluded Columns (and why)
- `patient_id` — unique identifier (not generalizable; risk of memorization)
- `admission_date`, `discharge_date` — raw timestamps (not used directly; only used to derive `length_of_stay`)

### Leakage Prevention
- `readmitted` is never included in features.
- Only `NUMERICAL_FEATURES + CATEGORICAL_FEATURES` are used as model inputs.
- Preprocessing is fit on training data only (`fit_transform` on train, `transform` on test/new data).

### Where X and y are defined (in code)

`src/data_preprocessing.py` separates features and target **before** splitting:

```python
from src.config import TARGET_COLUMN, ALL_FEATURES, EXCLUDED_COLUMNS

# Validate
assert TARGET_COLUMN not in ALL_FEATURES, "Target leaked into features!"

# Separate
X = df[ALL_FEATURES]
y = df[TARGET_COLUMN]
```

### Evaluation Metrics
- Precision, Recall, F1-score, ROC-AUC (accuracy is also reported but less reliable if the classes are imbalanced).

> **No real dataset?** Run `python generate_sample_dataset.py` to create 1 000 synthetic rows.

---

## Project Structure

```
S86-0326-MediWorks-Machine-Learning/

├── data/
│   ├── raw/
│   │   └── hospital_visits.csv
│   ├── processed/
│   └── external/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── persistence.py
│   └── predict.py
│
├── models/
│   ├── random_forest_readmission.pkl
│   └── preprocessing_pipeline.pkl
│
├── reports/
│   └── evaluation_metrics.json
│
├── logs/
│   └── experiment_log.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ML Pipeline Architecture

The pipeline follows a **modular and clean design**:

| Module                 | Purpose                   |
| ---------------------- | ------------------------- |
| config.py              | Configuration             |
| data_preprocessing.py  | Data cleaning & splitting |
| feature_engineering.py | Encoding & scaling        |
| train.py               | Model training            |
| evaluate.py            | Metrics                   |
| persistence.py         | Save/load                 |
| predict.py             | Prediction                |
| main.py                | Full pipeline             |

---

## Data Splitting Strategy

Before trusting any ML model, it must be tested on data it has **never seen**. If we train and test on the same rows, we measure memorization, not learning.

This project enforces a clean boundary between **training data** (used to learn patterns) and **test data** (used only for final evaluation).

### What we do (in this repo)

- **Split ratio:** 80% train / 20% test (`TEST_SIZE = 0.20`)
- **Reproducible split:** fixed seed (`RANDOM_STATE = 42`)
- **Stratified split:** preserves the class balance of `readmitted` in both sets
- **No leakage:** preprocessing is fitted only on training data

### Where it happens in code

- The split is performed in `src/data_preprocessing.py` inside `split_data()` using `train_test_split(..., stratify=y)`.
- The preprocessing pipeline is fitted only on training data in `main.py`:

```python
# split first
X_train, X_test, y_train, y_test = split_data(df_clean)

# fit preprocessing on training only
pipeline = build_preprocessing_pipeline(CATEGORICAL_COLS, NUMERICAL_COLS)
X_train_proc = pipeline.fit_transform(X_train)

# apply the same fitted preprocessing to the test set
X_test_proc = pipeline.transform(X_test)
```

### Common mistakes we avoid

- Scaling/encoding **before** splitting (leaks test statistics)
- Using the test set for hyperparameter tuning (inflates metrics)
- Forgetting stratification in classification (unstable evaluation on imbalanced data)

If you later work with time-series / chronological data, don’t shuffle — split by time (train on earlier dates, test on later dates).

---

## Installation & Setup

Before training:

* Numerical features checked using histograms, skewness, and boxplots
* Categorical features checked using value counts
* Target comparisons done to check predictive power

### Findings

* Skewed features → may need transformation
* Outliers → may need clipping
* Rare categories → grouped
* Useful features → show class difference

---

## Data Leakage Prevention

* Only training data uses `fit()`
* Test data uses `transform()`
* No preprocessing on full dataset

---

## Installation

```bash
git clone <repo-url>
cd project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Run Project

```bash
python main.py
```

---

## API Endpoints

* POST /upload
* GET /summary
* GET /peak-times
* GET /los
* GET /dept-load
* POST /predict-readmission

---

## Future Enhancements

* Real-time data integration
* Doctor availability prediction
* Alert system
* LSTM models

---

## License

MIT License

---

## Team

**MediWorks**
Healthcare + AI/ML
