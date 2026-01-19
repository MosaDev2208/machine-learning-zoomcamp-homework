import pickle
import xgboost as xgb
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Define the Input Schema (Data Validation)
# We list the columns we expect from the user (excluding the target 'maintenance_label')
class TurbineData(BaseModel):
    turbine_id: str
    rotor_speed_rpm: float
    wind_speed_mps: float
    power_output_kw: float
    gearbox_oil_temp_c: float
    generator_bearing_temp_c: float
    vibration_level_mmps: float
    ambient_temp_c: float
    humidity_pct: float

# 2. Load the Model
input_file = 'model.bin'
# We load both the DictVectorizer (dv) and the Model
with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

# 3. Create the App
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Wind Turbine Predictive Maintenance API is Live!"}

@app.post("/predict")
def predict(data: TurbineData):
    # Convert Pydantic object to Python dictionary
    turbine_dict = data.model_dump()
    
    # Transform data using the loaded DictVectorizer
    X = dv.transform([turbine_dict])
    
    # Create DMatrix (Required specifically for XGBoost)
    features = dv.get_feature_names_out()
    dmatrix = xgb.DMatrix(X, feature_names=features.tolist())
    
    # Predict
    probability = float(model.predict(dmatrix)[0])
    
    # Create response logic
    maintenance_needed = probability >= 0.5
    
    return {
        "maintenance_probability": probability,
        "maintenance_needed": maintenance_needed,
        "turbine_id": turbine_dict['turbine_id']
    }