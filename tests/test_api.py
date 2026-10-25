"""Automated tests for the FastAPI application."""

from fastapi.testclient import TestClient

from customer_churn_mlops.api import app


client = TestClient(app)


def test_health_check():
    """Verify that the health endpoint returns a successful response."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_churn():
    """Verify that the prediction endpoint returns valid results."""
    customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.5,
        "TotalCharges": 151.0,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()
    assert result["churn_prediction"] in [0, 1]
    assert 0.0 <= result["churn_probability"] <= 1.0

def test_predict_missing_fields():
    """Verify that incomplete customer data is rejected."""
    customer = {
        "gender": "Female",
        "tenure": 2,
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_negative_tenure():
    """Verify that negative tenure is rejected."""
    response = client.post(
        "/predict",
        json={"tenure": -1},
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "tenure"]
        and error["type"] == "greater_than_equal"
        for error in response.json()["detail"]
    )

def test_predict_tenure_above_maximum():
    """Verify that tenure above 72 months is rejected."""
    response = client.post(
        "/predict",
        json={"tenure": 73},
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "tenure"]
        and error["type"] == "less_than_equal"
        for error in response.json()["detail"]
    )

def test_predict_negative_monthly_charges():
    """Verify that negative monthly charges are rejected."""
    response = client.post(
        "/predict",
        json={"MonthlyCharges": -50},
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "MonthlyCharges"]
        and error["type"] == "greater_than_equal"
        for error in response.json()["detail"]
    )


def test_predict_negative_total_charges():
    """Verify that negative total charges are rejected."""
    response = client.post(
        "/predict",
        json={"TotalCharges": -100},
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "TotalCharges"]
        and error["type"] == "greater_than_equal"
        for error in response.json()["detail"]
    )


def test_predict_invalid_senior_citizen():
    """Verify that SeniorCitizen values above 1 are rejected."""
    response = client.post(
        "/predict",
        json={"SeniorCitizen": 2},
    )

    assert response.status_code == 422
    assert any(
        error["loc"] == ["body", "SeniorCitizen"]
        and error["type"] == "less_than_equal"
        for error in response.json()["detail"]
    )