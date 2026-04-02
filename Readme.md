# MEDILENS — AI-Powered Hospital Visit Analytics & Resource Optimization System

**Project Name:** MEDILENS
**Team Name:** MediWorks

MEDILENS is an AI and Machine Learning-based hospital analytics platform that analyses patient admission and discharge data to identify peak admission times, calculate average length of stay (LOS), detect overloaded departments, and **predict patient readmission risk** for better staffing and resource planning.

---

## Table of Contents

* Overview
* Problem Statement
* Proposed Solution
* Key Features
* System Workflow
* Tech Stack
* Dataset Format
* Project Structure
* ML Pipeline Architecture
* Installation & Setup
* Running the Pipeline
* API Endpoints
* Future Enhancements
* License

---

## Overview

Hospitals generate large volumes of patient visit data every day — admissions, discharges, department transfers, and bed occupancy. Most hospitals still depend on manual reporting and administrator intuition for staffing and resource decisions.

MEDILENS solves this using AI-driven analytics and Machine Learning forecasting to support data-driven hospital resource planning, including a **30-day readmission prediction model** built on a fully modular, reusable Python ML pipeline.

---

## Problem Statement

Hospitals generate vast amounts of patient admission and discharge data, yet administrators often rely on intuition for staffing decisions. This leads to:

* Staff shortage during peak hours
* Department overcrowding
* Poor bed availability planning
* Delayed discharge management
* Increased waiting time

---

## Proposed Solution

MEDILENS provides an AI-based hospital visit analytics system that:

* Identifies peak admission times
* Calculates average LOS
* Detects overloaded departments
* Predicts 30-day readmission risk
* Forecasts bed occupancy trends
* Provides staffing recommendations

---

## Key Features

### Hospital Analytics

* Admission trend analysis
* Peak time detection
* Department-wise analysis

### LOS Analysis

* Overall LOS
* Department-wise LOS

### ML Predictions

* Readmission prediction
* Admission forecasting

### Modular Pipeline

* Clean and reusable architecture
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

* Data preprocessing
* Problem type identification (classification vs regression)
* Feature engineering
* Model training
* Evaluation
* Prediction

---

## Problem Definition Before Training

Before training, MEDILENS now automatically inspects the target variable and
creates a formal problem-definition report.

For the current dataset (`readmitted`), the pipeline identifies:

* Task type: Classification
* Subtype: Binary classification
* Class distribution and imbalance ratio
* Recommended metrics and model families

Generated artifact:

* `reports/problem_definition.json`

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

* Real-time data
* Doctor prediction
* Alerts system
* LSTM models

---

## License

MIT License

---

## Team

**MediWorks**
Healthcare + AI/ML
