from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

class InputData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    logging.info(f"Received data: {data.dict()}")
    prediction = model.predict(df)[0]
    logging.info(f"Prediction result: {prediction}")
    return {"prediction": int(prediction)}
