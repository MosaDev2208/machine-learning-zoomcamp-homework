import pickle
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn

with open('pipeline_v1.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

class Client(BaseModel):
    lead_source: str = Field(..., description="Source of lead")
    number_of_courses_viewed: int = Field(..., ge=0)
    annual_income: float = Field(..., ge=0.0)

class PredictResponse(BaseModel):
    probability: float
    will_convert: bool

app = FastAPI(title="Lead Scoring API", version="1.0.0")

def predict_single(features: dict) -> float:
    probability = pipeline.predict_proba([features])[0, 1]
    return float(probability)

@app.post("/predict", response_model=PredictResponse)
def predict(client: Client) -> PredictResponse:
    features = client.model_dump()
    probability = predict_single(features)
    return PredictResponse(probability=probability, will_convert=probability >= 0.5)

@app.get("/")
def read_root():
    return {"service": "Lead Scoring API", "status": "running", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
