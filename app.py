from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import pandas as pd
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Logging setup
logging.basicConfig(level=logging.INFO)

# Prometheus metrics
REQUEST_COUNT = Counter('api_request_count', 'Total API Requests', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('api_request_latency_seconds', 'API Request latency', ['method', 'endpoint'])
ERROR_COUNT = Counter('api_error_count', 'Total API Errors', ['method', 'endpoint'])

# Input data schema
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

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = None
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        ERROR_COUNT.labels(request.method, request.url.path).inc()
        logging.error(f"Unhandled error: {e}")
        raise e
    finally:
        resp_time = time.time() - start_time
        REQUEST_LATENCY.labels(request.method, request.url.path).observe(resp_time)
        REQUEST_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
    return response

@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
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

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
