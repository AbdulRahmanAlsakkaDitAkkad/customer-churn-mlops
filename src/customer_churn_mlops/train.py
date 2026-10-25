"""Model training utilities for customer churn prediction."""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from customer_churn_mlops.preprocessing import (
    build_model_pipeline,
    clean_total_charges,
    split_features_target,
    split_train_test,
)

DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/logistic_regression.joblib")
mlflow.set_experiment("customer-churn-prediction")


def load_data() -> pd.DataFrame:
    """Load and clean the customer churn dataset."""
    df = pd.read_csv(DATA_PATH)
    df = clean_total_charges(df)

    return df


def train_model():
    """Train a Logistic Regression model on customer churn data."""
    df = load_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = build_model_pipeline()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def save_model(model, path=MODEL_PATH):
    """Save a trained machine learning pipeline to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate the trained model on the test dataset."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }

    return metrics


if __name__ == "__main__":
    with mlflow.start_run(run_name="logistic-regression-baseline"):
        model, X_test, y_test = train_model()
        save_model(model)
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="model")
        print(f"Model saved to: {MODEL_PATH}")
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.named_steps["classifier"].get_params())

        print("Logistic Regression Evaluation Results")
        print("--------------------------------------")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
