# Predictive Maintenance Optimization for Aircraft Engines

## Project Overview

An AI-driven predictive maintenance system for aircraft engines based on the NASA C-MAPSS (FD001) dataset.

The project combines **Remaining Useful Life (RUL) prediction** with **maintenance scheduling optimization** to reduce unexpected failures, control maintenance costs, and improve aircraft availability.

## Objectives

- Predict aircraft engine Remaining Useful Life (RUL).
- Optimize maintenance schedules.
- Reduce maintenance costs and unexpected failures.
- Improve aircraft availability and operational efficiency.
- Compare XGBoost and CNN approaches for RUL prediction.

## Methodology

### 1. Data Collection

NASA C-MAPSS (FD001) was used to collect aircraft engine sensor measurements and RUL data.

### 2. Data Preprocessing

The sensor data was prepared through:

- Handling missing values.
- Removing unnecessary sensors.
- Clipping outliers.
- Smoothing sensor signals.
- Normalizing features.

### 3. Feature Engineering

Rolling statistics and degradation-based features were created to capture engine health trends over time.

### 4. RUL Prediction

An **XGBoost regression model** was used to estimate the Remaining Useful Life (RUL) of aircraft engines.

A **CNN approach** was also used for model comparison.

### 5. Maintenance Optimization

An **Integer Linear Programming (ILP)** model was used to generate optimized maintenance schedules while considering:

- Safety constraints
- Capacity constraints
- Budget constraints
- Resource constraints

## Technologies

- Python
- XGBoost
- CNN
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Integer Linear Programming (ILP)
- HiGHS Solver

## My Contributions

My main contribution was the **Data Preprocessing** stage, including:

- Cleaning and preparing the NASA C-MAPSS (FD001) dataset.
- Handling missing values.
- Removing unnecessary sensors.
- Clipping outliers.
- Smoothing sensor signals.
- Normalizing features.
- Generating the cleaned dataset for subsequent project stages.

## Project Files

- `app-123.py` — Main application.
- `GP preprocessing.py` — Data preprocessing implementation.
- `CMAPSS_Dataset.xlsx` — Cleaned dataset generated during preprocessing.
- `Documentation.pdf` — Project documentation, methodology, experiments, analysis, and results.
- `Graduation presentation final.pptx` — Final project presentation.

## Academic Information

**Project:** Predictive Maintenance Optimization for Aircraft Engines  
**Project Type:** Graduation Project  
**University:** Cairo University  
**Faculty:** Computers and Artificial Intelligence

## Authors

Graduation Project Team — Cairo University
