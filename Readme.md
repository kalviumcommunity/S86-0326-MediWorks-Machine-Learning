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

| Column          | Description     |
| --------------- | --------------- |
| patient_id      | Unique ID       |
| admission_date  | Admission date  |
| discharge_date  | Discharge date  |
| department      | Department name |
| gender          | Gender          |
| admission_type  | Type            |
| bed_type        | Bed category    |
| age             | Age             |
| length_of_stay  | LOS             |
| num_procedures  | Procedures      |
| num_medications | Medications     |
| num_diagnoses   | Diagnoses       |
| readmitted      | Target          |

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
