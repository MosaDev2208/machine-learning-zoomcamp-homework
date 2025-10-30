"""
Lead Scoring Prediction Service
Module 5: ML Zoomcamp Deployment
Based on Workshop: Telco Churn Prediction

This module provides a FastAPI service for predicting lead conversion probability
using a pre-trained scikit-learn pipeline (DictVectorizer + LogisticRegression).
"""

import pickle
from typing import Literal
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn


# ============================================================================
# 1. LOAD PRE-TRAINED PIPELINE (Model initialization)
# ============================================================================

print("Loading pre-trained pipeline...")
with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

print("✅ Pipeline loaded successfully!")


# ============================================================================
# 2. PYDANTIC MODELS FOR DATA VALIDATION (Input/Output schemas)
# ============================================================================

class Client(BaseModel):
    """Lead/Client input data model - mirrors workshop Customer model"""
    lead_source: str = Field(..., description="Source of lead (e.g., organic_search, paid_ads)")
    number_of_courses_viewed: int = Field(..., ge=0, description="Number of courses viewed")
    annual_income: float = Field(..., ge=0.0, description="Annual income in USD")
    
    class Config:
        schema_extra = {
            "example": {
                "lead_source": "organic_search",
                "number_of_courses_viewed": 4,
                "annual_income": 80304.0
            }
        }


class PredictResponse(BaseModel):
    """API response model with prediction results"""
    probability: float = Field(..., description="Probability of conversion (0-1)")
    will_convert: bool = Field(..., description="Binary prediction: True if prob >= 0.5")
    
    class Config:
        schema_extra = {
            "example": {
                "probability": 0.534,
                "will_convert": True
            }
        }


# ============================================================================
# 3. PREDICTION HELPER FUNCTION
# ============================================================================

def predict_single(features: dict) -> float:
    """
    Predict conversion probability for a single lead
    
    This function mimics the workshop's pattern:
    - Takes a dict of features
    - Passes through the pre-trained pipeline
    - Returns probability of positive class
    
    Args:
        features: Dictionary with lead features
        
    Returns:
        Probability of conversion (float between 0 and 1)
    """
    probability = pipeline.predict_proba([features])[0, 1]
    return float(probability)


# ============================================================================
# 4. CREATE FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="Lead Scoring API",
    description="ML Zoomcamp Module 5 - Lead Conversion Prediction Service",
    version="1.0.0"
)


# ============================================================================
# 5. DEFINE API ENDPOINTS
# ============================================================================

@app.post("/predict", response_model=PredictResponse)
def predict(client: Client) -> PredictResponse:
    """
    Predict lead conversion probability
    
    Endpoint logic:
    1. Validate input using Pydantic (automatic type checking)
    2. Convert model to dict
    3. Call prediction function
    4. Return structured response
    
    Args:
        client: Validated Client object with lead features
        
    Returns:
        PredictResponse with probability and binary prediction
    """
    features = client.model_dump()
    probability = predict_single(features)
    
    return PredictResponse(
        probability=probability,
        will_convert=probability >= 0.5
    )


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "service": "Lead Scoring API",
        "status": "running",
        "docs": "/docs",
        "version": "1.0.0"
    }


# ============================================================================
# 6. RUN SERVER
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
