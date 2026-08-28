# Predictive Maintenance Optimization for Aircraft Engines

## Project Overview

A graduation project developed at **Cairo University, Faculty of Computers and Artificial Intelligence**.

The project focuses on **predictive maintenance for aircraft engines** using sensor data and machine learning techniques to estimate the **Remaining Useful Life (RUL)** of engine units.

The project includes data preprocessing, predictive modeling, and an application for analyzing aircraft engine maintenance data.

## Project Objectives

The main objectives of the project are to:

- Process and prepare aircraft engine sensor data for analysis.
- Estimate the Remaining Useful Life (RUL) of aircraft engines.
- Apply machine learning techniques for predictive maintenance.
- Analyze engine degradation patterns using sensor measurements.
- Support maintenance decisions by predicting potential engine failures before they occur.

## Dataset

The project uses aircraft engine degradation data based on the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

The dataset contains sensor measurements collected over engine operating cycles and is used to model engine degradation and estimate Remaining Useful Life.

The processed dataset used in the project is included in:

data/CMAPSS_Dataset.xlsx
Data Preprocessing

The preprocessing stage prepares the engine sensor data for predictive modeling.

The preprocessing workflow includes preparing and transforming the available engine operating-cycle and sensor measurements into a cleaned dataset suitable for subsequent analysis and modeling.

The preprocessing implementation is provided in:

GP preprocessing.py
Predictive Maintenance

Predictive maintenance aims to estimate the remaining operating time of an aircraft engine before failure.

The project uses historical sensor observations and engine operating cycles to identify degradation behavior and support Remaining Useful Life (RUL) estimation.

The main application implementation is provided in:

app-123.py
Project Structure
aircraft-engine-predictive-maintenance/
│
├── app-123.py
├── GP preprocessing.py
├── requirements.txt
├── README.md
│
├── data/
│   └── CMAPSS_Dataset.xlsx
│
├── documentation/
│   └── Documentation.pdf
│
└── presentation/
    └── Graduation presentation final (2).pptx
Files Description
app-123.py

Main Python application used for the project implementation and analysis.

GP preprocessing.py

Python implementation of the data preprocessing stage used to prepare the aircraft engine dataset.

requirements.txt

Contains the Python dependencies required to run the project.

CMAPSS_Dataset.xlsx

Cleaned and processed dataset generated from the preprocessing stage.

Documentation.pdf

Project documentation describing the project methodology, analysis, implementation, and results.

Graduation presentation final (2).pptx

Final graduation project presentation.

Technologies
Python
NumPy
Pandas
Matplotlib
Machine Learning
Predictive Maintenance
Data Preprocessing
RUL Estimation
Project Workflow
Raw Aircraft Engine Data
          ↓
    Data Preprocessing
          ↓
    Cleaned Dataset
          ↓
   Feature Preparation
          ↓
 Predictive Maintenance
          ↓
      RUL Estimation
          ↓
 Maintenance Analysis
Results

The project demonstrates how aircraft engine sensor data can be processed and analyzed to support predictive maintenance and Remaining Useful Life estimation.

The detailed experimental results, analysis, and project findings are available in:

documentation/Documentation.pdf
Academic Information

Project: Predictive Maintenance Optimization for Aircraft Engines
Project Type: Graduation Project
University: Cairo University
Faculty: Computers and Artificial Intelligence

Authors

Graduation Project Team
Cairo University

Documentation

For detailed methodology, implementation details, experiments, and results, refer to the project documentation included in the documentation folder.
