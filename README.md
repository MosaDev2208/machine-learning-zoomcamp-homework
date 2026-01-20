# Machine Learning Zoomcamp Portfolio

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![ML Zoomcamp](https://img.shields.io/badge/DataTalks%20Club-ML%20Zoomcamp%202024-orange)](https://datatalks.club/courses/2024-online-ml-zoomcamp.html)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-4d6eff?style=flat&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Comprehensive Machine Learning Engineering Portfolio** — A complete hands-on journey from ML fundamentals to production-ready deployments, covering regression, classification, ensemble methods, neural networks, and real-world MLOps projects.

---

## 🎓 About This Repository

This repository contains the complete coursework and capstone projects from the [DataTalks.Club Machine Learning Zoomcamp](https://datatalks.club/courses/2024-online-ml-zoomcamp.html) — a rigorous, project-based machine learning program designed to teach practical ML engineering skills used in industry.

**What You'll Find:**
- ✅ 8 comprehensive learning modules with hands-on exercises
- ✅ 2 production-grade capstone projects with complete documentation
- ✅ Real datasets from Kaggle with practical problem statements
- ✅ Best practices in model development, evaluation, and deployment
- ✅ Professional visualizations and detailed analysis
- ✅ Docker containerization and API development
- ✅ Complete source code with explanations

---

## 📚 Course Structure & Modules

### **Module 1: Introduction to Machine Learning** 
**Directory:** `01-intro/`

Learn the foundational concepts and frameworks that guide all ML projects.

**Topics Covered:**
- ML vs Rule-Based Systems — Understanding when to use ML
- Supervised Learning Basics — Core concepts and terminology
- CRISP-DM Methodology — Industry-standard project framework
- Model Selection — Choosing the right algorithm
- Environment Setup — Python, libraries, and dependencies

**Notebooks:**
- `numpy_practice.ipynb` — NumPy fundamentals for numerical computing
- `linear_algebra_practice.ipynb` — Matrix operations and linear transformations
- `pandas_practice.ipynb` — Data manipulation and exploration
- `homework_1.ipynb` — Applied exercises

**Skills Demonstrated:**
✅ Array operations and broadcasting  
✅ Matrix algebra (dot products, transposes, inverses)  
✅ Data frame manipulation and aggregation  
✅ Statistical analysis and visualization  

---

### **Module 2: Machine Learning for Regression**
**Directory:** `02-regression/`

Build a predictive model for continuous targets using linear regression techniques.

**Topics Covered:**
- Linear Regression from Scratch — Understanding the math
- Feature Engineering — Creating and selecting features
- Exploratory Data Analysis — Understanding your data
- Regularization Techniques — Preventing overfitting (Ridge, Lasso)
- Model Validation — Train/test splits and cross-validation

**Notebooks:**
- `homework_2.ipynb` — Car price prediction project

**Skills Demonstrated:**
✅ Simple and multiple linear regression  
✅ Feature scaling and normalization  
✅ Model evaluation metrics (RMSE, MAE, R²)  
✅ Identifying and handling overfitting/underfitting  

---

### **Module 3: Machine Learning for Classification**
**Directory:** `03-classification/`

Create a classification system to predict discrete categories.

**Topics Covered:**
- Logistic Regression — Binary and multi-class classification
- Feature Importance & Selection — Identifying key predictors
- Categorical Variable Encoding — Handling non-numeric features
- Model Interpretation — Understanding model decisions
- Decision Trees — Tree-based classification

**Notebooks:**
- `homework_3.ipynb` — Customer churn prediction

**Skills Demonstrated:**
✅ Binary and multi-class classification  
✅ Probability calibration  
✅ Feature selection techniques  
✅ Model interpretation and explainability  

---

### **Module 4: Evaluation Metrics for Classification**
**Directory:** `04-evaluation/`

Master the art of properly evaluating and comparing classification models.

**Topics Covered:**
- Classification Metrics — Accuracy, Precision, Recall, F1-Score
- ROC Curves & AUC — Threshold-independent evaluation
- Confusion Matrices — Understanding prediction errors
- Cross-Validation — Robust model evaluation
- Class Imbalance Handling — Techniques for skewed datasets
- Hyperparameter Tuning — Grid search and random search

**Notebooks:**
- `homework_4.ipynb` — Model comparison and selection

**Skills Demonstrated:**
✅ Multi-metric evaluation frameworks  
✅ Stratified k-fold cross-validation  
✅ ROC-AUC curve generation and interpretation  
✅ Imbalanced classification handling  
✅ Hyperparameter optimization  

---

### **Module 5: Deploying Machine Learning Models**
**Directory:** `05-deployment/`

Transform trained models into production-ready web services.

**Topics Covered:**
- Model Serialization — Saving and loading models (Pickle)
- FastAPI Web Services — Building REST APIs
- Pydantic Validation — Input validation and type checking
- Docker Containerization — Reproducible deployments
- Cloud Deployment — Lambda, Cloud Run, or container services
- Testing & Monitoring — Ensuring reliability

**Key Files:**
- `train.py` — Model training and serialization
- `predict.py` — FastAPI prediction service
- `test.py` — Comprehensive test suite
- `Dockerfile` — Container configuration
- `requirements.txt` — Dependency management

**Skills Demonstrated:**
✅ REST API development with FastAPI  
✅ Model deployment as microservices  
✅ Docker containerization  
✅ Input validation and error handling  
✅ API documentation (Swagger/OpenAPI)  

**Quick Start:**
```bash
cd 05-deployment
pip install -r requirements.txt
python train.py && python predict.py
```

---

### **Module 6: Decision Trees & Ensemble Learning**
**Directory:** `06-trees/` (covered in homework exercises)

Master tree-based models and ensemble techniques for superior predictions.

**Topics Covered:**
- Decision Trees — Understanding tree-based models
- Random Forest — Ensemble of decision trees
- Gradient Boosting — XGBoost fundamentals
- Feature Importance — Tree-based feature analysis
- Hyperparameter Tuning — Optimizing tree parameters
- Model Comparison — When to use which algorithm

**Skills Demonstrated:**
✅ Tree model training and evaluation  
✅ Ensemble method advantages  
✅ Feature importance extraction  
✅ Class imbalance handling in tree models  

---

### **Module 8: Neural Networks & Deep Learning**
**Directory:** `08-deep-learning/`

Introduction to deep learning with neural networks and CNNs.

**Topics Covered:**
- Neural Network Fundamentals — Layers, activation functions, backpropagation
- PyTorch Framework — Tensor operations and autograd
- TensorFlow & Keras — High-level deep learning API
- Convolutional Neural Networks (CNNs) — Image classification
- Transfer Learning — Leveraging pre-trained models
- Model Optimization — Training efficiency and hardware acceleration

**Dataset:** Hair texture classification (curly vs. straight images)

**Project Structure:**
- `homework.py` — CNN implementation
- `data/train/` — Training images (curly, straight)
- `data/test/` — Test images
- `hair_cnn_model.pth` — Saved model weights

**Skills Demonstrated:**
✅ Neural network architecture design  
✅ Convolutional layer operations  
✅ Data augmentation for images  
✅ Transfer learning with pre-trained models  
✅ Model training and optimization  

---

## 🏆 Capstone Projects

### **Capstone Project: Wind Turbine Predictive Maintenance**
**Directory:** `capstone-project/`

An end-to-end MLOps project predicting wind turbine failures to enable proactive maintenance and reduce costly downtime.

**🎯 Business Impact:**
- Reduce unplanned downtime by **30-50%**
- Minimize maintenance costs through predictive scheduling
- Extend turbine lifespan and improve operational efficiency

**📊 Dataset:** Kaggle Wind Turbine SCADA data (~10,000 records)
- Features: Rotor speed, wind speed, power output, temperatures, vibration
- Target: Binary classification (maintenance needed: yes/no)

**🛠️ Tech Stack:**
- Model: XGBoost with scikit-learn preprocessing
- API: FastAPI with Uvicorn
- Modern tooling: uv for dependency management
- Containerization: Docker for reproducibility

**📈 Model Performance:**
- Accuracy: 87%
- Precision: 82%
- Recall: 79%
- F1-Score: 0.80

**Key Features:**
✅ Professional README with badges and documentation  
✅ Interactive Jupyter notebook for experimentation  
✅ Production-ready Python training script  
✅ FastAPI REST API with Swagger UI  
✅ Docker containerization for deployment  
✅ Modern dependency management with uv  
✅ Comprehensive visualizations and analysis  

**Quick Start:**
```bash
cd capstone-project
uv sync
uv run python train.py
docker build -t wind-turbine-maintenance .
docker run -p 8000:8000 wind-turbine-maintenance
```

📖 **Full Documentation:** See [capstone-project/README.md](capstone-project/README.md)

---

### **Midterm Project: Industrial Machine Predictive Maintenance**
**Directory:** `midterm-project/`

A production-grade ML system predicting industrial machine failures using advanced classification techniques on real-world imbalanced data.

**🎯 Business Impact:**
- Prevent catastrophic equipment failures
- Optimize maintenance scheduling and costs
- Improve operational safety and efficiency

**📊 Dataset:** AI4I 2020 Predictive Maintenance Dataset (10,000 samples)
- Features: Temperature, speed, torque, tool wear, product type
- Target: Binary classification (machine failure: yes/no)
- Challenge: Severe class imbalance (3.39% failures)

**🛠️ Tech Stack:**
- Model: XGBoost with class weight balancing
- Preprocessing: scikit-learn (scaling, encoding)
- API: FastAPI with Pydantic validation
- Testing: Comprehensive test suite
- Containerization: Docker for production deployment

**📈 Model Performance:**
- Accuracy: 98.15%
- Precision: 69.62%
- Recall: 80.88% (prioritizes catching failures)
- ROC-AUC: 97.74%

**Key Features:**
✅ 10 professional visualizations (EDA + model evaluation)  
✅ Complete exploratory data analysis  
✅ Model comparison across 4 algorithms  
✅ Feature importance analysis  
✅ Confusion matrix and ROC curves  
✅ Production API with validation  
✅ Docker deployment ready  
✅ Comprehensive test coverage  

**Visualizations Included:**
- Target distribution (class imbalance)
- Failure types breakdown
- Feature correlation analysis
- Temperature, speed, and torque analysis
- Tool wear patterns
- Model comparison (ROC curves)
- Feature importance ranking
- Confusion matrix heatmap
- Performance metrics comparison

**Quick Start:**
```bash
cd midterm-project
pip install -r requirements.txt
python train.py
python predict.py
python test.py
```

📖 **Full Documentation:** See [midterm-project/README.md](midterm-project/README.md)

---

## 🛠️ Technology Stack

### Core ML & Data Science
| Tool | Version | Purpose |
|------|---------|---------|
| [Python](https://www.python.org/) | 3.12 | Programming language |
| [NumPy](https://numpy.org/) | ≥1.24 | Numerical computing |
| [Pandas](https://pandas.pydata.org/) | ≥2.0 | Data manipulation |
| [scikit-learn](https://scikit-learn.org/) | ≥1.3 | ML algorithms |
| [XGBoost](https://xgboost.readthedocs.io/) | ≥2.0 | Gradient boosting |
| [Matplotlib](https://matplotlib.org/) | ≥3.8 | Visualization |
| [Seaborn](https://seaborn.pydata.org/) | ≥0.13 | Statistical plots |

### Production & Deployment
| Tool | Purpose |
|------|---------|
| [FastAPI](https://fastapi.tiangolo.com/) | REST API framework |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [Pydantic](https://pydantic-settings.readthedocs.io/) | Data validation |
| [Docker](https://www.docker.com/) | Containerization |
| [uv](https://github.com/astral-sh/uv) | Fast dependency manager |

### Deep Learning
| Tool | Purpose |
|------|---------|
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [TensorFlow](https://www.tensorflow.org/) | Alternative DL framework |
| [Jupyter](https://jupyter.org/) | Interactive notebooks |

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.12+
- pip or conda
- Docker (optional)
- Git

### Installation

**Clone the repository:**
```bash
git clone https://github.com/MosaDev2208/machine-learning-zoomcamp.git
cd machine-learning-zoomcamp
```

**For module work:**
```bash
cd 01-intro
jupyter notebook numpy_practice.ipynb
```

**For deployment module:**
```bash
cd 05-deployment
pip install -r requirements.txt
python train.py && python predict.py
```

**For capstone project:**
```bash
cd capstone-project
uv sync
uv run python train.py
docker build -t wind-turbine-maintenance .
docker run -p 8000:8000 wind-turbine-maintenance
```

**For midterm project:**
```bash
cd midterm-project
pip install -r requirements.txt
python train.py
python predict.py
```

---

## 📊 Project Metrics

### Midterm Project - Industrial Machine Maintenance
- **Model Accuracy:** 98.15%
- **ROC-AUC:** 97.74%
- **Recall (Failure Detection):** 80.88%
- **Visualizations:** 10 professional charts
- **API Response Time:** <50ms

### Capstone Project - Wind Turbine Maintenance
- **Model Accuracy:** 87%
- **ROC-AUC:** 0.93
- **API Response Time:** <100ms
- **Container Size:** ~500MB

---

## 📈 Learning Outcomes

By completing this portfolio, you'll master:

### Fundamentals
✅ Linear algebra and numerical computing  
✅ Data manipulation and exploration  
✅ Statistical analysis and visualization  

### Machine Learning
✅ Linear and logistic regression  
✅ Decision trees and ensemble methods  
✅ Imbalanced classification handling  
✅ Hyperparameter tuning and optimization  
✅ Feature engineering and selection  

### Production & MLOps
✅ REST API development with FastAPI  
✅ Model serialization and deployment  
✅ Docker containerization  
✅ Testing and validation strategies  
✅ Error handling and monitoring  

### Advanced Topics
✅ Deep learning with neural networks  
✅ Convolutional neural networks (CNNs)  
✅ Transfer learning techniques  
✅ Modern dependency management (uv)  

---

## 🏗️ Repository Structure

```
machine-learning-zoomcamp-homework/
│
├── 01-intro/                       # Module 1: ML Fundamentals
│   ├── numpy_practice.ipynb
│   ├── linear_algebra_practice.ipynb
│   ├── pandas_practice.ipynb
│   └── homework_1.ipynb
│
├── 02-regression/                  # Module 2: Regression
│   └── homework_2.ipynb
│
├── 03-classification/              # Module 3: Classification
│   └── homework_3.ipynb
│
├── 04-evaluation/                  # Module 4: Model Evaluation
│   └── homework_4.ipynb
│
├── 05-deployment/                  # Module 5: Deployment
│   ├── train.py
│   ├── predict.py
│   ├── test.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── 08-deep-learning/               # Module 8: Deep Learning
│   ├── homework.py
│   ├── hair_cnn_model.pth
│   └── data/
│
├── midterm-project/                # Midterm: Industrial ML System
│   ├── README.md
│   ├── train.py
│   ├── predict.py
│   ├── test.py
│   ├── notebooks/notebook.ipynb
│   ├── images/                     # 10 visualizations
│   ├── Dockerfile
│   └── requirements.txt
│
├── capstone-project/               # Capstone: Wind Turbine ML
│   ├── README.md
│   ├── train.py
│   ├── predict.py
│   ├── notebook.ipynb
│   ├── model.bin
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── uv.lock
│   └── images/
│
└── README.md                       # This file
```

---

## 🧪 Testing & Quality

### Module 5 (Deployment)
```bash
cd 05-deployment
python test.py          # Run test suite
```

### Midterm Project
```bash
cd midterm-project
python test.py          # Comprehensive test coverage
```

### All Notebooks
All Jupyter notebooks are executable and self-contained with clear cell progression.

---

## 📚 Course Curriculum Highlights

| Module | Focus | Difficulty | Projects |
|--------|-------|-----------|----------|
| **1: Intro** | Fundamentals | Beginner | NumPy, Pandas, Linear Algebra |
| **2: Regression** | Continuous Targets | Beginner-Intermediate | Car Price Prediction |
| **3: Classification** | Discrete Targets | Intermediate | Churn Prediction |
| **4: Evaluation** | Model Selection | Intermediate | Model Comparison |
| **5: Deployment** | Production | Intermediate-Advanced | REST API, Docker |
| **6: Trees** | Ensemble Methods | Advanced | XGBoost Tuning |
| **8: Deep Learning** | Neural Networks | Advanced | CNN, Image Classification |
| **Midterm** | Real-World Problem | Advanced | Industrial ML System |
| **Capstone** | End-to-End MLOps | Advanced | Production ML Pipeline |

---

## 🎯 Key Skills Demonstrated

### Technical
- ✅ Data preprocessing and feature engineering
- ✅ Model training and evaluation
- ✅ Hyperparameter optimization
- ✅ REST API development
- ✅ Docker containerization
- ✅ Deep learning frameworks
- ✅ Cloud-ready deployment

### Professional
- ✅ Problem framing and analysis
- ✅ Business impact quantification
- ✅ Clear documentation
- ✅ Code quality and testing
- ✅ Production-ready practices
- ✅ Performance optimization

---

## 📝 Documentation

Each major project includes comprehensive documentation:

- **capstone-project/README.md** — Wind turbine project details
- **midterm-project/README.md** — Industrial machine project details
- **05-deployment/README.md** — API deployment guide
- Inline code comments and docstrings throughout

---

## 🤝 Best Practices Demonstrated

✅ **Version Control** — Git with meaningful commit messages  
✅ **Reproducibility** — Locked dependencies, random seeds  
✅ **Testing** — Unit tests and integration tests  
✅ **Documentation** — READMEs, badges, API docs  
✅ **Code Quality** — Clean code, type hints, error handling  
✅ **Production Ready** — Docker, logging, validation  

---

## 📞 Support & Resources

### Official Documentation
- [DataTalks.Club ML Zoomcamp](https://datatalks.club/courses/2024-online-ml-zoomcamp.html)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Guide](https://xgboost.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Docs](https://docs.docker.com/)

### Related Communities
- [MLOps.community](https://mlops.community/)
- [Kaggle](https://www.kaggle.com/)
- [Stack Overflow - Machine Learning](https://stackoverflow.com/questions/tagged/machine-learning)

---

## 📄 License

This project is part of the DataTalks.Club Machine Learning Zoomcamp curriculum.

---

## 🙏 Acknowledgments

- **DataTalks.Club** — For comprehensive ML education and curriculum design
- **Kaggle** — For high-quality datasets
- **Open-source communities** — For XGBoost, FastAPI, scikit-learn, and all tools used

---

**Last Updated:** January 2026 | **Status:** Complete & Production Ready ✅  
**Repository:** [machine-learning-zoomcamp](https://github.com/MosaDev2208/machine-learning-zoomcamp)  
**Course:** [ML Zoomcamp 2024-2025](https://datatalks.club/courses/2024-online-ml-zoomcamp.html)  
**Author:** Mosa Richard Papo
