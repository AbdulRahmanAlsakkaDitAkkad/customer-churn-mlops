"""Compare machine learning models for customer churn prediction."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from customer_churn_mlops.preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    split_features_target,
    split_train_test,
)
from customer_churn_mlops.train import (
    evaluate_model,
    load_data,
)

RANDOM_FOREST_PARAM_DISTRIBUTIONS = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 5, 10, 20],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
}


def build_random_forest_pipeline() -> Pipeline:
    """Build a Random Forest classification pipeline."""
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                ),
            ),
        ]
    )
    return pipeline


def train_random_forest():
    """Train a Random Forest model on customer churn data."""
    df = load_data()

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = build_random_forest_pipeline()

    model.fit(X_train, y_train)

    return model, X_test, y_test


def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters using cross-validation."""
    pipeline = build_random_forest_pipeline()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=RANDOM_FOREST_PARAM_DISTRIBUTIONS,
        n_iter=10,
        scoring="roc_auc",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    return search


def compare_models():
    """Train and evaluate Logistic Regression and Random Forest."""
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    models = {
        "Logistic Regression": build_model_pipeline(),
        "Random Forest": build_random_forest_pipeline(),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        results[name] = evaluate_model(model, X_test, y_test)

    search = tune_random_forest(X_train, y_train)
    print("\nBest Random Forest Hyperparameters:")
    print(search.best_params_)
    print(f"Best Cross-Validation ROC-AUC: {search.best_score_:.4f}")
    results["Tuned Random Forest"] = evaluate_model(
        search.best_estimator_, X_test, y_test
    )

    return results


if __name__ == "__main__":
    results = compare_models()

    comparison_df = pd.DataFrame.from_dict(results, orient="index")
    comparison_df.index.name = "Model"

    print("\nModel Comparison Summary")
    print(comparison_df.round(4).to_string())
    comparison_df.to_csv("reports/model_comparison.csv", index=True)
    for model_name, metrics in results.items():
        print(f"\n{model_name} Evaluation Results")
        print("--------------------------------")

        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
