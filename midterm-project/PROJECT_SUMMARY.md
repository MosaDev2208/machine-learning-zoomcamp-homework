# Predictive Maintenance ML Project - Summary

## Project Overview
Machine learning system to predict industrial machine failures before they occur, enabling proactive maintenance scheduling.

## Dataset
- **Source**: AI4I 2020 Predictive Maintenance Dataset (Kaggle)
- **Size**: 10,000 samples
- **Features**: 7 sensor readings (temperature, speed, torque, tool wear, product type)
- **Target**: Binary classification (failure/no failure)
- **Challenge**: Highly imbalanced (3.39% failure rate, 28.5:1 ratio)

## Model Performance
- **Algorithm**: XGBoost Classifier
- **Accuracy**: 98.15%
- **Precision**: 69.62%
- **Recall**: 80.88%
- **F1-Score**: 74.83%
- **ROC-AUC**: 97.74%

## Technology Stack
- Python 3.12, XGBoost, Scikit-learn, Pandas, NumPy
- FastAPI, Uvicorn, Pydantic, Docker
- Matplotlib, Seaborn

## Deployment
✅ Trained model saved (models/model.pkl)
✅ FastAPI service (predict.py)
✅ Dockerized (Dockerfile)
✅ Tested (test.py)
✅ Documented (README.md)

## Business Impact
- Reduce unplanned downtime by 30-50%
- Optimize maintenance schedules
- Lower maintenance costs
- Extend equipment lifespan

## Author
ML Zoomcamp Midterm Project 2025
