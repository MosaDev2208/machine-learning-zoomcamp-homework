#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import numpy as np
import uvicorn

# Load models
print("Loading models...")
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Models loaded successfully!")

# Create FastAPI app
app = FastAPI(
    title="Predictive Maintenance API",
    description="Predict machine failures before they occur",
    version="1.0.0"
)

# Define input schema
class MaintenanceInput(BaseModel):
    type: str = Field(..., description="Product type: L (Low), M (Medium), or H (High)")
    air_temperature: float = Field(..., description="Air temperature in Kelvin")
    process_temperature: float = Field(..., description="Process temperature in Kelvin")
    rotational_speed: int = Field(..., description="Rotational speed in RPM")
    torque: float = Field(..., description="Torque in Nm")
    tool_wear: int = Field(..., description="Tool wear in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "M",
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 1500,
                "torque": 40.0,
                "tool_wear": 100
            }
        }

# Define output schema
class MaintenancePrediction(BaseModel):
    failure_prediction: int
    failure_probability: float
    risk_level: str
    recommendation: str

@app.get("/")
def read_root():
    return {
        "message": "Predictive Maintenance API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=MaintenancePrediction)
def predict(input_data: MaintenanceInput):
    try:
        # Validate type
        if input_data.type not in ['L', 'M', 'H']:
            raise HTTPException(status_code=400, detail="Type must be L, M, or H")
        
        # Encode type
        type_encoded = label_encoder.transform([input_data.type])[0]
        
        # Create feature array
        temp_diff = input_data.process_temperature - input_data.air_temperature
        power_factor = input_data.torque * input_data.rotational_speed
        
        features = np.array([[
            input_data.air_temperature,
            input_data.process_temperature,
            input_data.rotational_speed,
            input_data.torque,
            input_data.tool_wear,
            type_encoded,
            temp_diff,
            power_factor
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "Low"
            recommendation = "Continue normal operations. Schedule routine maintenance."
        elif probability < 0.7:
            risk_level = "Medium"
            recommendation = "Monitor closely. Plan preventive maintenance within 48 hours."
        else:
            risk_level = "High"
            recommendation = "URGENT: Schedule immediate maintenance to prevent failure."
        
        return MaintenancePrediction(
            failure_prediction=int(prediction),
            failure_probability=round(float(probability), 4),
            risk_level=risk_level,
            recommendation=recommendation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting Predictive Maintenance API...")
    print("API documentation available at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
