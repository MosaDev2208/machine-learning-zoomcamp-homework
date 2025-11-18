# Predictive Maintenance for Industrial Machines

## Problem Description

Industrial machinery requires regular maintenance to prevent unexpected failures that cause costly production downtime. This project builds a machine learning model to predict machine failures before they occur, enabling proactive maintenance scheduling.

The model predicts machine failure and identifies five specific failure modes: Tool Wear Failure, Heat Dissipation Failure, Power Failure, Overstrain Failure, and Random Failures.

## Dataset

Source: AI4I 2020 Predictive Maintenance Dataset (Kaggle)
Size: 10,000 samples (3.39% failure rate)
Features: Air temperature, Process temperature, Rotational speed, Torque, Tool wear, Product type

## Installation

pip install -r requirements.txt

## Usage

python train.py
python predict.py
python test.py

## Docker

docker build -t predictive-maintenance .
docker run -p 8000:8000 predictive-maintenance

## Author

ML Zoomcamp Midterm Project 2025
