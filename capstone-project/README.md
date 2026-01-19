# 🌬️ Wind Turbine Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Service-green)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)

## 📌 Problem Description
Wind turbines are expensive machinery operating in harsh conditions. Unexpected failures lead to costly downtime. 

**The Goal:** Build a Machine Learning system to predict if a turbine needs maintenance based on sensor data.

**The Solution:** A binary classification model:
* `0`: **Healthy** (No Action Needed)
* `1`: **Maintenance Required** (Predicts failure before it happens)

---

## 🏗️ Project Architecture

```mermaid
graph LR
    A[Raw Data] --> B(Cleaning & EDA)
    B --> C{XGBoost Model}
    C --> D[Save model.bin]
    D --> E[FastAPI Service]
    E --> F[Docker Container]