# Wind Turbine Predictive Maintenance 🌬️

Welcome to the **Wind Turbine Predictive Maintenance** project! This repository hosts a Machine Learning solution designed to predict equipment failures before they happen. The goal is to leverage sensor data to identify whether a turbine is healthy or requires maintenance, reducing costly downtime and improving energy reliability.

This project is built with a rigorous **MLOps pipeline**, ensuring reproducibility and scalability using modern tools like `uv`, Docker, and FastAPI.

## 📝 **Problem Description**

Wind turbines are expensive assets operating in harsh environments. Unexpected failures in components like the gearbox or generator can lead to significant financial loss and energy disruption.

### **Objective**
The primary objective of this project is to develop a **binary classification model** capable of predicting the status of a turbine:

1.  **Healthy (0)**: No action required.
2.  **Maintenance Required (1)**: The turbine shows signs of wear or imminent failure (aggregating "Suggested" and "Immediate" categories).

By automating this detection, operators can schedule repairs proactively rather than reactively.

## 📊 **Dataset**

The dataset is sourced from **Kaggle** and contains telemetry sensor data from wind turbines.

### **Dataset Structure**
Key features used for prediction include:
- **`rotor_speed_rpm`**: Rotational speed of the main rotor.
- **`power_output_kw`**: Current power generation output.
- **`vibration_level_mmps`**: Vibration intensity (a critical indicator of mechanical wear).
- **`gearbox_oil_temp_c`**: Temperature of the gearbox oil.
- **`ambient_temp_c`**: Environmental temperature context.

### **Target Distribution**
Below is the distribution of the target variable. We mapped the original multi-class target to a binary classification problem.

![Target Distribution](target_distribution.png)

## 🔧 **Tools & Technologies**

This project utilizes the following tools and technologies:

- **Language**: Python 3.12
- **Dependency Management**: `uv` (Modern, fast replacement for pip/pipenv)
- **Machine Learning**: XGBoost, Scikit-Learn
- **Web Application Framework**: FastAPI
- **Containerization**: Docker

## ✨ **Setup**

### **Prerequisites**
- Python 3.12
- Docker (for containerized deployment)
- `uv` (for dependency management)

### **Local Setup**

#### 1. **Clone the Repository**
```bash
git clone [https://github.com/YOUR_USERNAME/machine-learning-zoomcamp-homework.git](https://github.com/YOUR_USERNAME/machine-learning-zoomcamp-homework.git)
cd capstone-project

2. Set Up the Python Environment
We use uv for lightning-fast setup.

Install uv (if not installed):
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

Install dependencies:
uv sync

Activate the environment:
source .venv/bin/activate

🚀 Exploratory Data Analysis and Modeling
The data analysis and model training process are documented in the notebook.ipynb file.

Key Insights
We analyzed feature correlations to understand which sensors impact the maintenance status most.

Training the Model
The final training logic is exported to train.py. You can reproduce the training process by running:

python train.py

📁 Deployment
This project includes a fully containerized FastAPI service for real-time predictions.

Local Deployment with Docker
You can build and run the service locally using Docker.
1. Build the Image:
docker build -t turbine-prediction .

2. Run the Container:
docker run -it --rm -p 8000:8000 turbine-prediction

Testing the API
Once the container is running, you can test it via curl in a separate terminal:

curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "turbine_id": "T-123",
  "rotor_speed_rpm": 1500.0,
  "wind_speed_mps": 12.5,
  "power_output_kw": 2500.0,
  "gearbox_oil_temp_c": 45.0,
  "generator_bearing_temp_c": 50.0,
  "vibration_level_mmps": 0.8,
  "ambient_temp_c": 22.0,
  "humidity_pct": 60.0
}'


📸 Proof of Deployment
1. Container Startup Logs:
2. Prediction Response:

🎉 Acknowledgments
A special thanks to DataTalks.Club for their free Machine Learning Zoomcamp course. The knowledge and skills gained from this course were instrumental in the development of this project.

If you're interested in learning more about machine learning, I highly recommend checking out their course repository.