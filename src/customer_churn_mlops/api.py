"""FastAPI application for customer churn prediction."""

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
    customer_df = pd.DataFrame([customer.model_dump()])

    prediction = int(model.predict(customer_df)[0])
    probability = float(model.predict_proba(customer_df)[0, 1])

    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
    }