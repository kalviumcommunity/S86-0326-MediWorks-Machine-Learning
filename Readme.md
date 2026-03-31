# MEDILENS — AI-Powered Hospital Visit Analytics & Resource Optimization System

**Project Name:** MEDILENS  
**Team Name:** MediWorks  

MEDILENS is an AI and Machine Learning-based hospital analytics platform that analyzes patient admission and discharge data to identify peak admission times, calculate average length of stay (LOS), detect overloaded departments, and predict future patient inflow for better staffing and resource planning.

---

## Table of Contents

- Overview
- Problem Statement
- Proposed Solution
- Key Features
- System Workflow
- Tech Stack
- Dataset Format
- Project Structure
- Installation & Setup
  - Backend Setup (Python)
  - Frontend Setup (React + Vite)
- Running the Project
- API Endpoints (Sample)
- Future Enhancements
- License

---

## Overview

Hospitals generate large volumes of patient visit data every day, including admissions, discharges, department transfers, and bed occupancy. However, most hospitals still depend on manual reporting and administrator intuition for staffing and resource decisions.

This causes overcrowding, bed shortages, inefficient staff scheduling, and longer patient waiting times.

MEDILENS solves this problem using AI-driven analytics and Machine Learning forecasting to support data-driven hospital resource planning.

---

## Problem Statement

Hospitals generate vast amounts of patient admission and discharge data, yet administrators often rely on intuition for staffing decisions. This leads to:

- Staff shortage during peak hours
- Department overcrowding (Emergency, ICU, General Medicine)
- Poor bed availability planning
- Delayed discharge management
- Increased waiting time and reduced service quality

A smart system is needed to analyze hospital visit patterns and forecast future patient load.

---

## Proposed Solution

MEDILENS provides an AI-based hospital visit analytics system that:

- Identifies peak admission hours/days/months
- Calculates average length of stay (LOS)
- Detects departments facing consistent overload
- Predicts future admissions using Machine Learning
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
- Department load risk scoring
- Overcrowding trend visualization

### 4. Machine Learning Predictions
- Admission forecasting (next day/week/month)
- Overload prediction for departments
- Bed occupancy forecasting

### 5. Dashboard Visualization
- Interactive graphs and charts
- Admin-friendly UI
- Reports and insights generation

---

## System Workflow

1. Hospital visit data is uploaded (CSV) or fetched from database.
2. Backend cleans and preprocesses the data.
3. Analytics module generates insights such as peak times, LOS, and overload patterns.
4. ML models train on historical hospital visit records.
5. Prediction engine forecasts future admissions and department overload probability.
6. Results are displayed in the React dashboard.

---

## Tech Stack

### Frontend
- React.js (Vite)
- Tailwind CSS / Bootstrap
- Chart.js / Recharts

### Backend
- Python (Flask or FastAPI)
- REST API

### Machine Learning
- Python
- Pandas, NumPy
- Scikit-learn
- Prophet / ARIMA (optional for time series forecasting)

### Database (Optional)
- SQLite (for development)

---

## Dataset Format

The system uses hospital visit data with admission and discharge details.

### Recommended Columns

| Column Name       | Description |
|------------------|-------------|
| patient_id       | Unique patient identifier |
| admission_date   | Admission date |
| admission_time   | Admission time |
| discharge_date   | Discharge date |
| discharge_time   | Discharge time |
| department       | Department name (Emergency, ICU, etc.) |
| bed_type         | General / ICU / Private |
| age              | Patient age |
| gender           | Patient gender |
| diagnosis        | Diagnosis category (optional) |

### Derived Fields
- Length of Stay (LOS) = discharge_datetime - admission_datetime
- Admission hour/day/month extracted for peak analysis

---

## Project Structure

Recommended folder structure:

```
medilens/
│
├── backend/
│   ├── app.py
│   ├── models/
│   ├── routes/
│   ├── ml/
│   ├── utils/
│   ├── requirements.txt
│   └── dataset/
│
├── frontend/
│   ├── index.html
│   ├── vite.config.js
│   ├── package.json
│   └── src/
│       ├── components/
│       ├── pages/
│       ├── services/
│       └── App.jsx
│
└── README.md
```

---

## Installation & Setup

### Prerequisites
Make sure you have installed:

- Python 3.10+
- Node.js 18+
- npm
- Git

---

## Backend Setup (Python)

Go to backend folder:

```bash
cd backend
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run backend server:

```bash
python app.py
```

Backend will run on:

```
http://localhost:5000
```

---

## Frontend Setup (React + Vite)

Go to frontend folder:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Run Vite development server:

```bash
npm run dev
```

Frontend will run on:

```
http://localhost:5173
```

---

## Running the Project

### Step 1: Start Backend

```bash
cd backend
python app.py
```

### Step 2: Start Frontend

```bash
cd frontend
npm run dev
```

### Step 3: Upload Dataset
Upload hospital dataset CSV using the dashboard upload option.

### Step 4: View Analytics and Predictions
Dashboard will show peak admissions, LOS reports, department overload, and forecasts.

---

## API Endpoints (Sample)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | /api/upload | Upload hospital dataset |
| GET    | /api/summary | Overall summary report |
| GET    | /api/peak-times | Peak admission analysis |
| GET    | /api/los | Length of stay analytics |
| GET    | /api/dept-load | Department load report |
| GET    | /api/predict-admissions | Admission forecasting |
| GET    | /api/predict-overload | Overload prediction |

---

## Future Enhancements

- Real-time hospital data integration
- Doctor availability prediction
- Emergency patient inflow forecasting
- SMS/Email overload alerts
- Bed allocation recommendation engine
- Patient readmission prediction
- Deep learning forecasting using LSTM

---

## License

This project is licensed under the MIT License.

---

## Team

**Team Name:** MediWorks  
**Project:** MEDILENS  
**Domain:** Healthcare Analytics + AI/ML
