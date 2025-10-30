import pickle
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from fastapi import FastAPI
import uvicorn
import logging

# ===========================
# LOGGING SETUP
# ===========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("\n" + "=" * 80)
print("🚀 CHURN PREDICTION API - PRODUCTION DEPLOYMENT")
print("=" * 80 + "\n")

# ===========================
# PYDANTIC MODELS
# ===========================

class Customer(BaseModel):
    """
    Customer data model with strict validation and realistic example
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "female",
                "seniorcitizen": 0,
                "partner": "yes",
                "dependents": "no",
                "phoneservice": "no",
                "multiplelines": "no_phone_service",
                "internetservice": "dsl",
                "onlinesecurity": "no",
                "onlinebackup": "yes",
                "deviceprotection": "no",
                "techsupport": "no",
                "streamingtv": "no",
                "streamingmovies": "no",
                "contract": "month-to-month",
                "paperlessbilling": "yes",
                "paymentmethod": "electronic_check",
                "tenure": 1,
                "monthlycharges": 29.85,
                "totalcharges": 29.85
            }
        }
    )
    
    gender: Literal["male", "female"] = Field(..., description="Customer gender")
    seniorcitizen: Literal[0, 1] = Field(..., description="Senior citizen status (0=No, 1=Yes)")
    partner: Literal["yes", "no"] = Field(..., description="Has partner")
    dependents: Literal["yes", "no"] = Field(..., description="Has dependents")
    phoneservice: Literal["yes", "no"] = Field(..., description="Phone service")
    multiplelines: Literal["no", "yes", "no_phone_service"] = Field(..., description="Multiple phone lines")
    internetservice: Literal["dsl", "fiber_optic", "no"] = Field(..., description="Internet service type")
    onlinesecurity: Literal["no", "yes", "no_internet_service"] = Field(..., description="Online security service")
    onlinebackup: Literal["no", "yes", "no_internet_service"] = Field(..., description="Online backup service")
    deviceprotection: Literal["no", "yes", "no_internet_service"] = Field(..., description="Device protection service")
    techsupport: Literal["no", "yes", "no_internet_service"] = Field(..., description="Tech support service")
    streamingtv: Literal["no", "yes", "no_internet_service"] = Field(..., description="Streaming TV service")
    streamingmovies: Literal["no", "yes", "no_internet_service"] = Field(..., description="Streaming movies service")
    contract: Literal["month-to-month", "one_year", "two_year"] = Field(..., description="Contract type")
    paperlessbilling: Literal["yes", "no"] = Field(..., description="Paperless billing")
    paymentmethod: Literal["electronic_check", "mailed_check", "bank_transfer_(automatic)", "credit_card_(automatic)"] = Field(..., description="Payment method")
    tenure: int = Field(..., ge=0, description="Months as customer (0-72)")
    monthlycharges: float = Field(..., ge=0.0, description="Monthly billing amount ($)")
    totalcharges: float = Field(..., ge=0.0, description="Total billing amount ($)")

class PredictResponse(BaseModel):
    """Prediction response model"""
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn (0-1)")
    churn: bool = Field(..., description="Will customer churn? (True=Yes, False=No)")

# ===========================
# LOAD MODEL
# ===========================

print("📦 Loading trained model from model.bin...")
try:
    with open('model.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)
    logger.info("✅ Model loaded successfully!")
    print("✅ Model loaded successfully!\n")
except FileNotFoundError as e:
    logger.error("❌ model.bin not found! Make sure it's in the same directory.")
    print(f"❌ Error: {e}\n")
    raise
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    print(f"❌ Error: {e}\n")
    raise

# ===========================
# FASTAPI APP
# ===========================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict if a telecom customer will churn using Machine Learning",
    version="3.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# ===========================
# HELPER FUNCTIONS
# ===========================

def predict_single(customer_dict: dict) -> float:
    """
    Get churn probability from model
    
    ✅ CRITICAL FIX: Pass customer_dict as list to predict_proba
    sklearn's predict_proba expects array-like of shape (n_samples, n_features)
    NOT a single dictionary
    
    Args:
        customer_dict: Customer features dictionary
        
    Returns:
        float: Churn probability (0.0 to 1.0)
    """
    try:
        # ✅ CRITICAL FIX: Wrap in list [customer_dict]
        prediction = pipeline.predict_proba([customer_dict])
        
        # Get probability of class 1 (churn = True)
        churn_prob = prediction[0, 1]
        
        # Ensure it's a Python float (not numpy float64)
        return float(churn_prob)
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise

# ===========================
# API ENDPOINTS
# ===========================

@app.get("/")
def root():
    """Root endpoint - returns API information"""
    return {
        "message": "🎯 Customer Churn Prediction API v3.0",
        "status": "🟢 RUNNING",
        "docs": "📚 /docs",
        "model": "🤖 Logistic Regression",
        "training_data": "Telco Customer Churn Dataset"
    }

@app.get("/health")
def health_check():
    """Health check endpoint - returns API and model status"""
    return {
        "status": "🟢 healthy",
        "model": "🤖 loaded",
        "api_version": "3.0.0",
        "ready_for_predictions": True
    }

@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer) -> PredictResponse:
    """
    Predict if customer will churn
    
    COMPLETE FIX CHECKLIST:
    ✅ 1. Wraps customer dict in list for predict_proba
    ✅ 2. Extracts probability correctly (class 1)
    ✅ 3. Converts to Python float
    ✅ 4. Makes boolean decision with explicit bool()
    ✅ 5. Logs all details for monitoring
    ✅ 6. Uses realistic example in Swagger UI
    
    Args:
        customer: Customer data (validated by Pydantic)
        
    Returns:
        PredictResponse: Contains churn_probability and churn boolean
    """
    try:
        # Convert Pydantic model to dictionary
        customer_dict = customer.model_dump()
        
        # Log request details
        logger.info("=" * 80)
        logger.info("📊 PREDICTION REQUEST RECEIVED")
        logger.info("=" * 80)
        logger.info(f"  Customer Profile:")
        logger.info(f"    • Gender: {customer_dict['gender']}")
        logger.info(f"    • Tenure: {customer_dict['tenure']} months")
        logger.info(f"    • Monthly Charges: ${customer_dict['monthlycharges']:.2f}")
        logger.info(f"    • Total Charges: ${customer_dict['totalcharges']:.2f}")
        logger.info(f"    • Contract: {customer_dict['contract']}")
        logger.info(f"    • Internet Service: {customer_dict['internetservice']}")
        
        # Get probability from model
        probability = predict_single(customer_dict)
        
        # Make churn decision (>= 0.5 threshold)
        churn_decision = bool(probability >= 0.5)
        
        # Log prediction results
        logger.info(f"\n  🤖 Model Output:")
        logger.info(f"    • Churn Probability: {probability:.4f} ({probability*100:.2f}%)")
        logger.info(f"    • Churn Decision: {'🔴 YES (Will Churn)' if churn_decision else '🟢 NO (Will Stay)'}")
        
        # Business logic
        if churn_decision:
            logger.info(f"    • Recommendation: 💡 SEND RETENTION OFFER")
        else:
            logger.info(f"    • Recommendation: ✅ MAINTAIN RELATIONSHIP")
        
        logger.info("=" * 80 + "\n")
        
        return PredictResponse(
            churn_probability=probability,
            churn=churn_decision
        )
        
    except ValueError as ve:
        logger.error(f"❌ Validation error: {ve}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise

# ===========================
# RUN SERVER
# ===========================

if __name__ == "__main__":
    print("=" * 80)
    print("🌐 STARTING UVICORN SERVER")
    print("=" * 80)
    print("📍 API URL:        http://0.0.0.0:8080")
    print("📚 Docs URL:       http://localhost:8080/docs")
    print("💚 Health Check:   http://localhost:8080/health")
    print("🏠 Root Endpoint:  http://localhost:8080/")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
