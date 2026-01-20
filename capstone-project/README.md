# Wind Turbine Predictive Maintenance

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-4d6eff?style=flat&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/dependency%20manager-uv-4c8bf5)](https://github.com/astral-sh/uv)

**End-to-end MLOps project** — predicting wind turbine maintenance needs from sensor data using modern tooling.

## 🎯 Business Problem

Wind turbine downtime is extremely expensive — both in lost energy production and repair costs. Predictive maintenance allows operators to perform interventions **before** a failure occurs, significantly reducing unplanned outages and extending equipment lifespan.

This project implements a **binary classification model** that predicts whether a turbine requires maintenance (1 = Maintenance Needed, 0 = Healthy) based on real-time sensor readings.

## 📊 Dataset

Source: Kaggle wind turbine SCADA dataset (commonly used in predictive maintenance challenges)

**Key features** used for modeling:

- `rotor_speed_rpm` — Rotor rotational speed
- `wind_speed_mps` — Wind speed at hub height
- `power_output_kw` — Active power produced
- `gearbox_oil_temp_c` — Gearbox oil temperature
- `generator_bearing_temp_c` — Generator bearing temperature
- `vibration_level_mmps` — Vibration level
- `ambient_temp_c` — Outside air temperature
- `humidity_pct` — Relative humidity

Target: `maintenance_label` (0 = healthy, 1 = maintenance needed, 2 = critical — simplified to binary in modeling)

### Exploratory Data Analysis Highlights

**Class distribution** (highly imbalanced — typical for real maintenance problems):

![Target Distribution (Original)](images/target_distribution.png)

**Feature correlations** (strong relationships between power, wind speed, rotor speed, and temperatures):

![Feature Correlation Matrix](images/correlation_matrix.png)

## 🛠 Tech Stack

- **Language**: Python 3.12
- **Dependency & Environment Management**: [uv](https://github.com/astral-sh/uv) — ultra-fast modern replacement for pip + virtualenv
- **Modeling**: XGBoost + scikit-learn preprocessing
- **API Framework**: FastAPI
- **Containerization & Deployment**: Docker
- **Experimentation**: Jupyter notebook

## Project Structure

```text
capstone-project/
├── data/
│   └── data.csv
├── images/
│   ├── target_distribution.png
│   ├── correlation_matrix.png
│   ├── deployment_test1.png
│   └── deployment_test2.png
├── notebook.ipynb          # EDA + model experiments
├── train.py                # Training script → produces model.bin
├── predict.py              # FastAPI prediction service
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── README.md
└── .gitignore

 Quick Start (Local without Docker)

1. git clone https://github.com/YOUR-USERNAME/wind-turbine-predictive-maintenance.git
cd wind-turbine-predictive-maintenance/capstone-project

2. Install uv (if not already installed)
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
or follow https://github.com/astral-sh/uv#installation

3. Sync dependencies & create virtual environment
uv sync --frozen

4. Activate environment
source .venv/bin/activate      # Linux/macOS
# or
.venv\Scripts\activate         # Windows

5. Train the model
python train.py
→ creates model.bin

 Docker Deployment

1. Build the image
docker build -t turbine-predictor .

2. Run the container (maps port 8000)
docker run -it --rm -p 8000:8000 turbine-predictor
You should see Uvicorn startup logs:Docker startup logs

 API Testing
Send a POST request to /predict:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "turbine_id": "T-123",
    "rotor_speed_rpm": 1500.0,
    "wind_speed_mps": 12.5,
    "power_output_kw": 2500.0,
    "gearbox_oil_temp_c": 65.2,
    "generator_bearing_temp_c": 72.8,
    "vibration_level_mmps": 1.45,
    "ambient_temp_c": 18.3,
    "humidity_pct": 74.0
  }'

Example successful response:
API prediction response

 Model Output Format

{
  "turbine_id": "T-123",
  "maintenance_probability": 0.08123,
  "maintenance_needed": false
}

 Acknowledgments
 This project was completed as part of the DataTalks.Club Machine Learning Zoomcamp capstone module (2025 edition).Special thanks to Alexey Grigorev and the entire DTC community for the excellent course structure and support.

