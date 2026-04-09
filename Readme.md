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
* Feature Distribution Analysis
* Feature Type Definition
* Data Leakage Prevention
* Baseline Model Comparison
* Logistic Regression Classification Tutorial
* Feature Engineering Notes
* Installation & Setup
* Running the Pipeline
* API Endpoints
* Future Enhancements
* License
* Team

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

| Column          | Description    |
| --------------- | -------------- |
| patient_id      | Unique ID      |
| admission_date  | Admission date |
| discharge_date  | Discharge date |
| department      | Department     |
| gender          | Gender         |
| admission_type  | Type           |
| bed_type        | Bed            |
| age             | Age            |
| length_of_stay  | LOS            |
| num_procedures  | Procedures     |
| num_medications | Medications    |
| num_diagnoses   | Diagnoses      |
| readmitted      | Target         |

---

## Project Structure

```
S86-0326-MediWorks-Machine-Learning/
├── data/
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── logistic_regression_tutorial.py
│   ├── train.py
│   ├── evaluate.py
│   ├── persistence.py
│   └── predict.py
├── models/
├── reports/
├── logs/
├── main.py
├── requirements.txt
└── README.md
```

---

## ML Pipeline Architecture

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

## Feature Distribution Analysis

### Findings

* Skewed features → may need transformation
* Outliers → may need clipping
* Rare categories → grouped
* Useful features → show class difference

### Confirming No Data Leakage

All analysis is based only on raw data inspection. No preprocessing or model training was done before train-test split.

### Next Steps

* Handle skewness
* Handle outliers
* Balance categorical data
* Prepare preprocessing pipeline

---

## Feature Type Definition

### Target Variable

* **Column:** `readmitted`
* **Type:** Binary (0/1)

### Numerical Features

* age
* length_of_stay
* num_procedures
* num_medications
* num_diagnoses

### Categorical Features

* department
* gender
* admission_type
* bed_type
asdfghjkl;'

### Excluded Columns

* patient_id → identifier
* admission_date → raw timestamp
* discharge_date → data leakage

---

## Data Leakage Prevention

* Fit only on training data
* Transform test data
* No preprocessing on full dataset

---

## Baseline Model Comparison

### Purpose

Establish a minimum benchmark to ensure the ML model provides real value.

### Baseline Strategy

* Model: `DummyClassifier` (most_frequent)
* Always predicts majority class

### Key Insight

* Accuracy alone is misleading
* Precision, Recall, F1, ROC-AUC are essential
* Baseline ensures model usefulness

### Run Comparison

```bash
python run_baseline_comparison.py
```

---

## Logistic Regression Classification Tutorial

This repository now includes a complete, code-first Logistic Regression lesson for binary classification (no video script format).

### What This Covers

* Binary classification intuition and probability prediction
* Majority-class baseline using `DummyClassifier`
* Leakage-safe preprocessing + Logistic Regression pipeline
* Evaluation using Accuracy, F1, ROC-AUC, and classification report
* Cross-validation for stability checks
* Hyperparameter tuning for regularization strength (`C`)
* Coefficient interpretation in log-odds and odds ratio form

### Run the Tutorial

```bash
python -m src.logistic_regression_tutorial
```

### Why This Matters

* Logistic Regression is a strong first benchmark for classification.
* It is interpretable, fast, and often competitive with more complex models.
* It helps validate whether the dataset has real predictive signal before trying heavier models.

---

## Feature Engineering Notes

* Numerical features scaled using `StandardScaler`
* Categorical features encoded using One-Hot Encoding
* Train/test split done before preprocessing
* Pipeline reusable for prediction

---

## Installation & Setup

```bash
git clone <repo-url>
cd project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running the Pipeline

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
Healthcare analytics team behind MEDILENS.
