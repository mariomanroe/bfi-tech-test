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
    try:
        df = pd.DataFrame([data.dict()])

        # Rename kolom agar sesuai dengan nama fitur model
        df.columns = [
            "fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]

        logging.info(f"Received data: {data.dict()}")
        prediction = model.predict(df)[0]
        logging.info(f"Prediction result: {prediction}")
        return {"prediction": int(prediction)}
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}
