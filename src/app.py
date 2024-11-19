from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()
model = joblib.load('model.pkl')

# Input data schema
class HouseFeatures(BaseModel):
    size: float
    rooms: int

# Prediction endpoint
@app.post("/predict/")
async def predict(features: HouseFeatures):
    # Extract features and predict
    data = np.array([features.size, features.rooms]).reshape(1, -1)
    predicted_price = model.predict(data)[0]
    return {"predicted_price": predicted_price}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "House Price Predictor is running!"}
