
"""Unit tests for customer churn preprocessing."""

import pandas as pd

from customer_churn_mlops.preprocessing import (
    clean_total_charges,
    split_features_target,
    split_train_test,
    build_preprocessor,
    build_model_pipeline,
)
from sklearn.linear_model import LogisticRegression

def test_clean_total_charges_converts_strings():
    """Check that TotalCharges strings are converted to numbers."""
    df = pd.DataFrame({
        "tenure": [1, 2, 3],
        "TotalCharges": ["29.85", "188.95", "108.15"],
    })

    result = clean_total_charges(df)

    assert result["TotalCharges"].dtype == "float64"
    assert result["TotalCharges"].tolist() == [29.85, 188.95, 108.15]



def test_clean_total_charges_zero_tenure():
    """Check that missing charges are zero for new customers."""
    df = pd.DataFrame({
        "tenure": [0, 1, 0],
        "TotalCharges": [" ", "29.85", " "],
    })

    result = clean_total_charges(df)

    assert result["TotalCharges"].tolist() == [0.0, 29.85, 0.0]
    assert result["TotalCharges"].isna().sum() == 0



def test_clean_total_charges_preserves_other_missing_values():
    """Keep missing charges when customer tenure is not zero."""
    df = pd.DataFrame({
        "tenure": [0, 5, 2],
        "TotalCharges": [" ", " ", "100.50"],
    })

    result = clean_total_charges(df)

    assert result.loc[0, "TotalCharges"] == 0.0
    assert pd.isna(result.loc[1, "TotalCharges"])
    assert result.loc[2, "TotalCharges"] == 100.50


def test_clean_total_charges_does_not_modify_original():
    """Ensure cleaning does not change the original DataFrame."""
    df = pd.DataFrame({
        "tenure": [0, 1],
        "TotalCharges": [" ", "29.85"],
    })

    original_df = df.copy(deep=True)

    result = clean_total_charges(df)

    pd.testing.assert_frame_equal(df, original_df)
    assert result.loc[0, "TotalCharges"] == 0.0


def test_split_features_target():
    """Check that features and churn labels are separated correctly."""
    df = pd.DataFrame({
        "customerID": ["A001", "A002", "A003"],
        "tenure": [1, 12, 24],
        "Churn": ["Yes", "No", "Yes"],
    })

    X, y = split_features_target(df)

    assert "customerID" not in X.columns
    assert "Churn" not in X.columns
    assert X.columns.tolist() == ["tenure"]
    assert y.tolist() == [1, 0, 1]

    
def test_split_train_test_preserves_class_distribution():
    """Check train/test sizes and churn class proportions."""
    X = pd.DataFrame({"tenure": range(100)})
    y = pd.Series([0] * 75 + [1] * 25)

    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=0.2, random_state=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.mean() == 0.25
    assert y_test.mean() == 0.25


def test_split_train_test_is_reproducible():
    """Check that the same random state produces identical splits."""
    X = pd.DataFrame({"tenure": range(100)})
    y = pd.Series([0] * 75 + [1] * 25)

    first_split = split_train_test(X, y, random_state=42)
    second_split = split_train_test(X, y, random_state=42)

    
    for first, second in zip(first_split, second_split):
        if isinstance(first, pd.DataFrame):
            pd.testing.assert_frame_equal(first, second)
        else:
            pd.testing.assert_series_equal(first, second)



def test_build_preprocessor_has_expected_transformers():
    """Check that preprocessing includes all three feature groups."""
    preprocessor = build_preprocessor()

    transformer_names = [
        name for name, _, _ in preprocessor.transformers
    ]

    assert transformer_names == [
        "numeric",
        "binary",
        "categorical",
    ]


def test_build_model_pipeline_has_expected_steps():
    """Check that the model pipeline includes preprocessing and classification."""
    model = build_model_pipeline()

    assert list(model.named_steps.keys()) == [
        "preprocessor",
        "classifier",
    ]
    assert isinstance(
        model.named_steps["classifier"],
        LogisticRegression,
    )
