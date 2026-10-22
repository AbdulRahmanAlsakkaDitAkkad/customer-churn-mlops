
"""Model training utilities for customer churn prediction."""

import pandas as pd
from pathlib import Path

from customer_churn_mlops.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
    build_model_pipeline,
)


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")



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
    model, X_test, y_test = train_model()
    metrics = evaluate_model(model, X_test, y_test)

    print("Logistic Regression Evaluation Results")
    print("--------------------------------------")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
