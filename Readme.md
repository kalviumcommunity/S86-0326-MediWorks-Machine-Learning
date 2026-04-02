# MEDILENS — AI-Powered Hospital Visit Analytics & Resource Optimization System

**Project Name:** MEDILENS  
**Team Name:** MediWorks  

MEDILENS is an AI and Machine Learning-based hospital analytics platform that analyses patient
admission and discharge data to identify peak admission times, calculate average length of stay
(LOS), detect overloaded departments, and **predict patient readmission risk** for better
staffing and resource planning.

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
- [Installation & Setup](#installation--setup)
- [Running the Pipeline](#running-the-pipeline)
- [API Endpoints (Sample)](#api-endpoints-sample)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Overview

Hospitals generate large volumes of patient visit data every day — admissions, discharges,
department transfers, and bed occupancy.  Most hospitals still depend on manual reporting and
administrator intuition for staffing and resource decisions.

MEDILENS solves this using AI-driven analytics and Machine Learning forecasting to support
data-driven hospital resource planning, including a **30-day readmission prediction model**
built on a fully modular, reusable Python ML pipeline.

---

## Problem Statement

Hospitals generate vast amounts of patient admission and discharge data, yet administrators
often rely on intuition for staffing decisions.  This leads to:

- Staff shortage during peak hours  
- Department overcrowding (Emergency, ICU, General Medicine)  
- Poor bed availability planning  
- Delayed discharge management  
- Increased waiting time and reduced service quality  

---

## Proposed Solution

MEDILENS provides an AI-based hospital visit analytics system that:

- Identifies peak admission hours / days / months  
- Calculates average length of stay (LOS)  
- Detects departments facing consistent overload  
- **Predicts 30-day readmission risk** using a Random Forest classifier  
- Forecasts department load and bed occupancy trends  
- Generates staffing and resource planning recommendations  

---

## Key Features

### 1. Hospital Visit Analytics
- Hourly, daily, and monthly admission trend analysis  
- Peak admission time detection  
- Department-wise patient inflow analysis  
- Discharge trend analysis  

### 2. Length of Stay (LOS) Analysis
- Overall hospital average LOS calculation  
- Department-wise LOS calculation  
- Identification of long-stay patient patterns  

### 3. Department Overload Detection
- Detects consistently overloaded departments  
- Overcrowding trend visualisation  

### 4. Machine Learning Predictions
- **30-day readmission prediction** (core ML model)  
- Admission forecasting (next day / week / month)  
- Bed occupancy forecasting  

### 5. Modular ML Pipeline
- Clean separation of data loading, preprocessing, feature engineering,
  training, evaluation, persistence, and prediction  
- Each stage is encapsulated in its own function and module  
- Every function is documented, typed, and independently testable  

---

## System Workflow

```
Raw CSV  →  load_data()  →  validate_schema()  →  clean_data()
         →  split_data()
         →  build_preprocessing_pipeline().fit_transform(X_train)
                                          .transform(X_test)
         →  train_model()
         →  evaluate_model()
         →  save_artifacts()
                               ↓ (later)
               load_artifacts()  →  predict()  →  Output DataFrame
```

---

## Tech Stack

### Machine Learning
- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn (Random Forest, ColumnTransformer, OneHotEncoder, StandardScaler)  
- Joblib (artifact serialisation)  

### Frontend (planned)
- React.js (Vite)  
- Chart.js / Recharts  

### Backend (planned)
- Python (FastAPI)  
- REST API  

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
│
├── data/
│   ├── raw/
│   │   └── hospital_visits.csv          ← place your dataset here
│   └── processed/                       ← (reserved for future use)
│
├── models/
│   ├── random_forest_readmission.pkl    ← saved model  (generated by main.py)
│   └── preprocessing_pipeline.pkl      ← saved pipeline (generated by main.py)
│
├── reports/
│   └── evaluation_metrics.json         ← accuracy, F1, ROC-AUC, etc.
│
├── src/
│   ├── __init__.py                      ← makes src/ a Python package
│   ├── config.py                        ← all paths, column names, hyperparams
│   ├── data_preprocessing.py            ← load, validate, clean, split
│   ├── feature_engineering.py           ← drop IDs, build ColumnTransformer
│   ├── train.py                         ← fit RandomForest, return model
│   ├── evaluate.py                      ← compute metrics dict (no printing)
│   ├── persistence.py                   ← save / load artifacts with joblib
│   └── predict.py                       ← inference (transform, never fit)
│
├── main.py                              ← orchestration script (full pipeline)
├── generate_sample_dataset.py           ← create synthetic data for testing
├── requirements.txt
└── README.md
```

---

## ML Pipeline Architecture

The pipeline follows strict **single-responsibility** design:

| Module | Function(s) | Purpose |
|---|---|---|
| `config.py` | — | Central configuration (paths, columns, hyperparams) |
| `data_preprocessing.py` | `load_data`, `validate_schema`, `clean_data`, `split_data` | Ingest → validate → clean → split |
| `feature_engineering.py` | `drop_id_columns`, `build_preprocessing_pipeline` | Encode + scale features |
| `train.py` | `train_model` | Fit model, return artifact |
| `evaluate.py` | `evaluate_model` | Compute metrics dict |
| `persistence.py` | `save_artifacts`, `load_artifacts` | Joblib serialisation |
| `predict.py` | `predict` | Inference using saved artifacts |
| `main.py` | `main` | Orchestrates all steps in sequence |

### Key Design Decisions

1. **`fit_transform` only on training data** — the preprocessing pipeline is fitted
   exclusively on `X_train`. `X_test` and new inference data use `.transform()`.
   This prevents data leakage.

2. **Functions return values, never print** — core functions return dicts, DataFrames,
   or model objects. Only the orchestration layer (`main.py`) prints to the console.

3. **Explicit `random_state` everywhere** — all stochastic operations accept a
   `random_state` parameter, defaulting to the centralised value in `config.py`.

4. **Centralised configuration** — no hardcoded paths or magic numbers inside
   functions. Everything flows from `src/config.py`.

---

## Installation & Setup

### Prerequisites

- Python 3.12
- Git

### Environment Setup

Create a dedicated virtual environment in the project root and install pinned dependencies.

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/S86-0326-MediWorks-Machine-Learning.git
cd S86-0326-MediWorks-Machine-Learning

# 2. Create the virtual environment
python -m venv venv

# 3. Activate the environment
# Windows PowerShell
venv\Scripts\Activate.ps1
# Windows Command Prompt
venv\Scripts\activate.bat
# macOS / Linux
source venv/bin/activate

# 4. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Reproduce from scratch

```bash
git clone https://github.com/<your-username>/S86-0326-MediWorks-Machine-Learning.git
cd S86-0326-MediWorks-Machine-Learning
python -m venv venv
source venv/bin/activate          # macOS / Linux
# or venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Notes

- The `venv/` folder is excluded from version control in `.gitignore`.
- This setup uses a local Python environment so global packages do not affect project execution.
- This branch is prepared for pull request submission and contains the reproducible environment setup updates.
- To exit the environment, run `deactivate`.
- If Python is not available as `python`, use `python3` on your system.

---

## Running the Pipeline

```bash
# Step 1 — (Optional) Generate synthetic dataset if you don't have a real one
python generate_sample_dataset.py

# Step 2 — Run the full training pipeline
python main.py
```

Expected output:

```
============================================================
  MEDILENS — Hospital Readmission Prediction Pipeline
============================================================

[1/7] Loading raw data ...
      Loaded 1,000 rows × 13 columns.
[2/7] Validating schema ...
      Schema OK — all required columns present.
[3/7] Cleaning data ...
[4/7] Splitting data (test_size=0.2, random_state=42) ...
      Training rows : 800
      Test rows     : 200
[5/7] Dropping ID columns and building preprocessing pipeline ...
[6/7] Training Random Forest model ...
[7/7] Evaluating model on held-out test set ...

  ── Evaluation Results ──────────────────────────────────
  accuracy    : 0.7650
  precision   : 0.6200
  recall      : 0.5800
  f1          : 0.5993
  roc_auc     : 0.8100
  ────────────────────────────────────────────────────────

  Model saved    → models/random_forest_readmission.pkl
  Pipeline saved → models/preprocessing_pipeline.pkl
  Metrics report → reports/evaluation_metrics.json

  Pipeline complete. MEDILENS model is ready.
```

---

## API Endpoints (Sample)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload hospital dataset |
| `GET`  | `/api/summary` | Overall summary report |
| `GET`  | `/api/peak-times` | Peak admission analysis |
| `GET`  | `/api/los` | Length of stay analytics |
| `GET`  | `/api/dept-load` | Department load report |
| `POST` | `/api/predict-readmission` | 30-day readmission prediction |

---

## Future Enhancements

- Real-time hospital data integration  
- Doctor availability prediction  
- Emergency patient inflow forecasting  
- SMS / Email overload alerts  
- Bed allocation recommendation engine  
- Patient readmission prediction using LSTM  
- FastAPI backend wiring `predict()` to a REST endpoint  

---

## License

This project is licensed under the MIT License.

---

## Team

**Team Name:** MediWorks  
**Project:** MEDILENS  
**Domain:** Healthcare Analytics + AI/ML
