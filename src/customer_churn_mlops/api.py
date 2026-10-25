"""FastAPI application for customer churn prediction."""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import time

import joblib
import pandas as pd
from fastapi import FastAPI

from customer_churn_mlops.schemas import CustomerInput
from customer_churn_mlops.train import MODEL_PATH

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Machine learning API for predicting customer churn.",
    version="0.1.0",
)

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health_check():
    """Check whether the API is running."""
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    """Predict customer churn using the trained model."""
    start_time = time.perf_counter()
    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0, 1])
    logger.info(
        "Prediction completed: churn_prediction=%s, churn_probability=%.4f",
        prediction,
        probability,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.info("Prediction latency: %.2f ms", latency_ms)

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
    }
