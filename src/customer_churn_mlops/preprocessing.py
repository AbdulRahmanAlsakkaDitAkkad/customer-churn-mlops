"""Data preprocessing utilities for customer churn prediction.

This module provides reusable functions for cleaning customer data,
preparing features, and building machine learning preprocessing pipelines.

Preprocessing steps that learn from data must be fitted only on the
training set to prevent data leakage.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

BINARY_FEATURES = [
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric.

    Fill missing TotalCharges with zero for customers with zero tenure.
    Preserve other missing values for further investigation.
    """
    cleaned_df = df.copy()

    cleaned_df["TotalCharges"] = pd.to_numeric(
        cleaned_df["TotalCharges"], errors="coerce"
    )

    # New customers with zero tenure have no accumulated charges.
    zero_tenure_missing = cleaned_df["tenure"].eq(0) & cleaned_df["TotalCharges"].isna()

    cleaned_df.loc[zero_tenure_missing, "TotalCharges"] = 0.0

    return cleaned_df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate customer features from the binary churn target."""
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"Yes": 1, "No": 0})

    return X, y


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified training and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """Build a preprocessing transformer for customer features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def build_model_pipeline() -> Pipeline:
    """Build a pipeline combining preprocessing and Logistic Regression."""
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    return model_pipeline
