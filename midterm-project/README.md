# Predictive Maintenance for Industrial Machines

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-4d6eff?style=flat&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat&logo=scikit-learn)](https://scikit-learn.org/)

**Production-ready ML pipeline** — Predicting industrial machine failures using advanced ensemble learning techniques to enable proactive maintenance and minimize costly downtime.

---

## 🎯 Business Problem

Unexpected industrial machinery failures are **extraordinarily expensive**, causing:
- **Production downtime:** $1000s per hour in lost output
- **Emergency repairs:** 3-5x more costly than planned maintenance
- **Equipment damage:** Extended failure duration causes cascading failures
- **Safety risks:** Unplanned failures can compromise worker safety

**Predictive maintenance** enables operators to:
✅ Schedule maintenance **before** failures occur  
✅ Reduce unplanned downtime by **30-50%**  
✅ Optimize maintenance budgets and resource allocation  
✅ Extend equipment lifespan through proactive care  
✅ Improve workplace safety and operational efficiency  

This project builds a **high-accuracy binary classification model** that predicts machine failures based on real-time sensor data, allowing maintenance teams to act before catastrophic failures occur.

---

## 📊 Dataset

**Source:** [AI4I 2020 Predictive Maintenance Dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) (Kaggle)

**Dataset Size:** 10,000 operational records with 7 sensor features

**Class Distribution:** Highly imbalanced (3.39% failures, 96.61% normal operation)

**Key Features Used for Modeling:**

| Feature | Description | Unit | Range |
|---------|-------------|------|-------|
| `air_temperature_k` | Air temperature in production environment | Kelvin | 295.3-304.5 K |
| `process_temperature_k` | Temperature during machine operation | Kelvin | 305.3-313.8 K |
| `rotational_speed_rpm` | Machine spindle rotational speed | RPM | 1168-2886 |
| `torque_nm` | Motor torque during operation | Newton-meters | 3.8-76.6 Nm |
| `tool_wear_min` | Cumulative tool wear over time | Minutes | 0-258 |
| `product_type` | Type of product being manufactured | Categorical | A, B, C, D, E |

**Target Variable:** `machine_failure` (Binary: 0 = No Failure, 1 = Failure)

**Failure Modes:**
- **Tool Wear Failure (TWF):** Tool exceeds maximum wear threshold
- **Heat Dissipation Failure (HDF):** Cooling system malfunction causes overheating
- **Power Failure (PWF):** Unexpected power loss or electrical issues
- **Overstrain Failure (OSF):** Machine components exceed stress limits
- **Random Failure (RNF):** Unpredictable equipment malfunction

---

## 📈 Exploratory Data Analysis (EDA)

The dataset exhibits **severe class imbalance** with 96.61% negative samples (healthy machines) and only 3.39% positive samples (machine failures). This is typical for real-world predictive maintenance problems where failures are rare events.

### Key Insights:
- **Temperature correlation:** Process temperature and air temperature are highly correlated with failure events
- **Wear accumulation:** Tool wear increases failure risk significantly
- **Speed factor:** Higher rotational speeds correlate with certain failure types
- **Product type:** Different product types exhibit different failure patterns

### Mitigation Strategies:
- Used XGBoost's native handling of imbalanced datasets
- Applied class weighting to penalize minority class misclassification
- Evaluated using ROC-AUC (better for imbalanced data than accuracy)

---

## 🛠️ Tech Stack

- **Language:** [Python 3.12](https://www.python.org/downloads/) — Modern, performant Python with type safety
- **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/) — Gradient boosting with native imbalanced data handling
- **Preprocessing:** [scikit-learn](https://scikit-learn.org/) — Data scaling, encoding, train/test splits
- **Data Processing:** [Pandas](https://pandas.pydata.org/) + [NumPy](https://numpy.org/) — Data manipulation and numerical computing
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) — Modern, async REST API framework
- **Server:** [Uvicorn](https://www.uvicorn.org/) — ASGI web server
- **Visualization:** [Matplotlib](https://matplotlib.org/) + [Seaborn](https://seaborn.pydata.org/) — EDA plotting
- **Containerization:** [Docker](https://www.docker.com/) — Reproducible, portable deployment
- **Validation:** [Pydantic](https://pydantic-settings.readthedocs.io/) — Data validation and settings management

---

## 📁 Project Structure

```
midterm-project/
├── data/
│   └── ai4i2020.csv                # Raw dataset (10,000 samples, 7 features)
├── notebooks/
│   └── notebook.ipynb              # EDA and model development notebook
├── train.py                        # Model training script (outputs models/model.pkl)
├── predict.py                      # FastAPI application for serving predictions
├── test.py                         # Comprehensive test suite
├── Dockerfile                      # Docker container configuration
├── requirements.txt                # Python dependencies
├── .dockerignore                   # Docker build exclusions
├── .gitignore                      # Git exclusions
├── PROJECT_SUMMARY.md              # High-level project overview
├── README.md                       # This file (detailed documentation)
└── models/
    └── model.pkl                   # Trained XGBoost model (generated after train.py)
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.12 installed on your system
- pip or conda package manager
- Docker (optional, for containerized deployment)

### Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/MosaDev2208/machine-learning-zoomcamp-homework.git
   cd machine-learning-zoomcamp-homework/midterm-project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or with conda:
   ```bash
   conda create -n predictive-maintenance python=3.12
   conda activate predictive-maintenance
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "import xgboost; import fastapi; print('✅ All dependencies installed successfully')"
   ```

---

## 🚂 Model Training

Train the XGBoost classifier on the AI4I 2020 dataset:

```bash
python train.py
```

**What this script does:**
1. Loads the dataset from `data/ai4i2020.csv`
2. Performs exploratory data analysis and preprocessing
3. Encodes categorical features (product type)
4. Scales numerical features using StandardScaler
5. Splits data into 80% training, 20% testing
6. Trains an XGBoost classifier with class weight balancing
7. Evaluates performance on test set
8. Saves the trained model to `models/model.pkl`

**Expected output:**
```
Loading dataset...
Data shape: (10000, 7)
Training set size: 8000
Test set size: 2000
Failure rate: 3.39%

Training XGBoost model...
Model training complete!

Model Performance:
  Accuracy:  98.15%
  Precision: 69.62%
  Recall:    80.88%
  F1-Score:  74.83%
  ROC-AUC:   97.74%

Model saved to: models/model.pkl
```

### Model Hyperparameters:
```python
XGBClassifier(
    max_depth=7,
    learning_rate=0.1,
    n_estimators=200,
    random_state=42,
    scale_pos_weight=28.5,      # Handle class imbalance
    eval_metric='logloss'
)
```

---

## 🚀 Deployment

### Option 1: FastAPI Local Deployment

Start the prediction API locally:

```bash
python predict.py
```

The API will start on `http://localhost:8000`

**Interactive API Documentation:**
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Option 2: Docker Deployment (Recommended for Production)

#### Build the Docker image:
```bash
docker build -t predictive-maintenance .
```

#### Run the container:
```bash
docker run -p 8000:8000 predictive-maintenance
```

The API will be accessible at `http://localhost:8000`

#### Stop the container:
```bash
docker ps                    # Get container ID
docker stop <container_id>
```

---

## 🧪 Testing the API

Once the service is running (locally or in Docker), make prediction requests:

### Using curl

**Single Machine Prediction:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 298.5,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1500,
    "torque_nm": 42.8,
    "tool_wear_min": 120,
    "product_type": "M"
  }'
```

**Expected Response:**
```json
{
  "prediction": 0,
  "failure_probability": 0.08,
  "status": "healthy",
  "message": "Machine is operating normally. No maintenance needed."
}
```

**Machine with High Failure Risk:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "air_temperature_k": 300.2,
    "process_temperature_k": 312.5,
    "rotational_speed_rpm": 2800,
    "torque_nm": 70.0,
    "tool_wear_min": 240,
    "product_type": "A"
  }'
```

**Expected Response:**
```json
{
  "prediction": 1,
  "failure_probability": 0.92,
  "status": "at_risk",
  "message": "⚠️ Failure risk detected! Schedule maintenance immediately."
}
```

### Using Python

```python
import requests

url = "http://localhost:8000/predict"

machine_data = {
    "air_temperature_k": 298.5,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1500,
    "torque_nm": 42.8,
    "tool_wear_min": 120,
    "product_type": "M"
}

response = requests.post(url, json=machine_data)
result = response.json()

print(f"Status: {result['status']}")
print(f"Failure Probability: {result['failure_probability']:.2%}")
print(f"Message: {result['message']}")
```

### Batch Predictions from CSV:

```python
import pandas as pd
import requests

# Load machine data
df = pd.read_csv('machine_data.csv')

# Make predictions for each machine
predictions = []
for _, row in df.iterrows():
    response = requests.post(
        "http://localhost:8000/predict",
        json=row.to_dict()
    )
    predictions.append(response.json())

# Display results
results_df = pd.DataFrame(predictions)
print(results_df)
```

---

## ✅ Running Tests

Execute the comprehensive test suite:

```bash
python test.py
```

**Test Coverage:**
- Data loading and validation
- Model training and serialization
- Prediction API functionality
- Input validation and error handling
- Performance metrics verification
- Docker container health checks

**Expected output:**
```
Running test suite...

✓ Test 1: Data loading
✓ Test 2: Model training
✓ Test 3: Model inference
✓ Test 4: API endpoint
✓ Test 5: Input validation
✓ Test 6: Error handling

All tests passed! ✅
```

---

## 📚 Model Performance Analysis

### Overall Metrics

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| **Accuracy** | 98.15% | 98 out of 100 predictions are correct |
| **Precision** | 69.62% | 70% of predicted failures are actual failures |
| **Recall** | 80.88% | Model catches 81% of actual failures |
| **F1-Score** | 74.83% | Balanced measure between precision and recall |
| **ROC-AUC** | 97.74% | Excellent discrimination between classes |

### Why These Metrics Matter

**Precision (69.62%):** Minimizes false alarms that disrupt production schedules  
**Recall (80.88%):** Catches most real failures before they occur (critical for safety)  
**ROC-AUC (97.74%):** Model has excellent ability to rank predictions

### Class Imbalance Handling

The dataset has a **28.5:1 ratio** (healthy:failure). To address this:
- ✅ Used `scale_pos_weight=28.5` in XGBoost
- ✅ Evaluated using ROC-AUC instead of accuracy
- ✅ Optimized for recall to catch more failures
- ✅ Applied StandardScaler for feature normalization

---

## 🔄 Development Workflow

### Exploratory Data Analysis

For interactive model development and analysis, use Jupyter notebook:

```bash
jupyter notebook notebooks/notebook.ipynb
```

The notebook contains:
- Data loading and exploration
- Feature engineering and visualization
- Model training experiments
- Performance evaluation
- Prediction examples

### Reproducing Results

To ensure reproducible results across runs:

```bash
# Set random seeds
export PYTHONHASHSEED=42
export CUDA_LAUNCH_BLOCKING=1

# Train model
python train.py
```

All random states are fixed in the code:
- `random_state=42` in sklearn
- `seed=42` in XGBoost
- `np.random.seed(42)`

---

## 📋 API Endpoints

### `/` (GET)
Welcome endpoint with API information

```bash
curl http://localhost:8000/
```

### `/predict` (POST)
Make a failure prediction for a machine

**Request Body:**
```json
{
  "air_temperature_k": 298.5,
  "process_temperature_k": 308.6,
  "rotational_speed_rpm": 1500,
  "torque_nm": 42.8,
  "tool_wear_min": 120,
  "product_type": "M"
}
```

**Response:**
```json
{
  "prediction": 0,
  "failure_probability": 0.08,
  "status": "healthy",
  "message": "Machine is operating normally. No maintenance needed."
}
```

### `/health` (GET)
Health check endpoint

```bash
curl http://localhost:8000/health
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'xgboost'` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: models/model.pkl` | Train the model first with `python train.py` |
| `Port 8000 already in use` | Use different port: `docker run -p 8001:8000 predictive-maintenance` |
| `Docker build fails` | Ensure Docker is installed and running: `docker --version` |
| `API returns 422 Unprocessable Entity` | Check JSON request format and field names |
| `Low recall on new data` | Model may need retraining; check for data drift |

---

## 📝 Key Learnings & Future Enhancements

### Key Achievements
✅ Built production-ready MLOps pipeline with FastAPI and Docker  
✅ Achieved 98.15% accuracy on imbalanced industrial dataset  
✅ Implemented high recall (80.88%) to catch failures effectively  
✅ Created comprehensive API with interactive documentation  
✅ Deployed containerized solution for easy scalability  

### Future Enhancements
- 🔮 **Real-time monitoring:** Add streaming predictions from sensor feeds
- 🔮 **Model monitoring:** Implement drift detection and automated retraining
- 🔮 **Explainability:** Add SHAP values for failure mode interpretation
- 🔮 **Cloud deployment:** Deploy to AWS Lambda, Google Cloud Run, or Azure Functions
- 🔮 **Multi-model ensemble:** Combine XGBoost with neural networks
- 🔮 **Web dashboard:** Build real-time monitoring UI with Streamlit or Dash
- 🔮 **A/B testing:** Compare model versions in production
- 🔮 **Feature store:** Implement centralized feature management for scaling

---

## 📄 License

This project is part of the [DataTalks.Club Machine Learning Zoomcamp](https://datatalks.club/courses/2024-online-ml-zoomcamp.html).

---

## 🙏 Acknowledgments

- **DataTalks.Club Machine Learning Zoomcamp** — For comprehensive curriculum and guidance
- **Kaggle & Stephan Matzka** — For the AI4I 2020 Predictive Maintenance Dataset
- **XGBoost, FastAPI, scikit-learn communities** — For excellent open-source tools
- **Industrial ML community** — For best practices in predictive maintenance

---

## 📧 Contact & Support

For questions, issues, or suggestions, please open an issue in the [GitHub repository](https://github.com/MosaDev2208/machine-learning-zoomcamp-homework).

---

**Last Updated:** January 2026 | **Status:** Production Ready ✅  
**Project Type:** MLOps Midterm Project | **Zoomcamp:** 2024-2025
