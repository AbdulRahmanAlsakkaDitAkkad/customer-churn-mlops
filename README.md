
[![CI](https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops/actions/workflows/ci.yml)

# Customer Churn Prediction & MLOps Platform

An end-to-end machine learning engineering portfolio project that predicts customer churn and demonstrates reproducible ML workflows, backend API development, containerization, experiment tracking, automated testing, and cloud deployment.

**Project status:** Deployed and operational. Core development, testing, and cloud deployment are complete. Final portfolio documentation and presentation are in progress.

## Live Demo

**Interactive API documentation:**  
https://customer-churn-mlops-89f8.onrender.com/docs

**Health endpoint:**  
https://customer-churn-mlops-89f8.onrender.com/health

**GitHub repository:**  
https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops

The application is deployed as a Docker-based web service on Render.

> **Note:** The service uses Render's free hosting tier, which may spin down during inactivity. The first request after an idle period can take approximately one minute while the service starts.

## Project Overview

Customer churn prediction helps businesses identify customers who may discontinue their services.

This project uses the IBM Telco Customer Churn dataset to develop a machine learning system that estimates the probability of customer churn.

The project demonstrates practical skills in:

- Exploratory data analysis and data quality assessment
- Data preprocessing and feature engineering
- Supervised machine learning and model evaluation
- Model comparison and hyperparameter tuning
- Reproducible scikit-learn pipelines
- REST API development with FastAPI
- API input validation with Pydantic
- Docker containerization
- Cloud deployment on Render
- Experiment tracking with MLflow
- Automated testing with pytest
- Code quality enforcement with Ruff
- Continuous integration using GitHub Actions
- Prediction logging and latency measurement

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python 3.12 |
| Data analysis | pandas, NumPy |
| Machine learning | scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Experiment tracking | MLflow |
| Backend API | FastAPI, Pydantic, Uvicorn |
| Testing | pytest, FastAPI TestClient |
| Code quality | Ruff |
| Package management | uv |
| Containerization | Docker |
| Cloud hosting | Render |
| CI | GitHub Actions |
| Development environment | VS Code, Ubuntu on WSL2 |
| Version control | Git, GitHub |

## Dataset

**Dataset:** IBM Telco Customer Churn

**Source:**  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains:

- 7,043 customer records
- 21 original columns
- Customer demographics
- Subscription and internet service information
- Contract and payment details
- Customer tenure and charges
- Customer churn labels

The dataset is stored at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The CSV file is included in the repository to support reproducible model training, including training in GitHub Actions.

## Exploratory Data Analysis

The exploratory analysis examined customer characteristics, churn distribution, contract types, payment methods, and customer tenure.

### Key Findings

| Metric | Finding |
|---|---|
| Total customers | 7,043 |
| Overall churn rate | 26.54% |
| Month-to-month contract churn | 42.71% |
| One-year contract churn | 11.27% |
| Two-year contract churn | 2.83% |
| Fiber optic customer churn | 41.89% |
| Electronic check customer churn | 45.29% |
| Median tenure of churned customers | 10 months |
| Median tenure of retained customers | 38 months |

Customers with month-to-month contracts showed substantially higher churn rates than customers with longer contracts.

Customers who churned also tended to have shorter tenure.

These findings describe statistical associations and should not be interpreted as causal relationships.

### Data Quality

The analysis identified 11 missing `TotalCharges` values, all associated with customers with zero tenure.

These values are handled during preprocessing.

No duplicate customer IDs or fully duplicated rows were identified.

**EDA notebook:**  
[01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)

## Data Preprocessing

A reusable preprocessing pipeline was implemented using scikit-learn.

The preprocessing workflow includes:

1. Converting `TotalCharges` to numeric values
2. Handling missing charges for customers with zero tenure
3. Separating input features and target labels
4. Removing customer identifiers from model inputs
5. Performing a stratified train/test split
6. Standardizing numerical features
7. One-hot encoding categorical features
8. Combining preprocessing and classification into one pipeline

### Train/Test Split

| Dataset | Percentage | Records |
|---|---|---|
| Training | 80% | 5,634 |
| Testing | 20% | 1,409 |

A fixed random seed (`random_state=42`) supports reproducibility.

**Implementation:**  
`src/customer_churn_mlops/preprocessing.py`

## Machine Learning Models

Three model configurations were evaluated:

1. Logistic Regression
2. Random Forest
3. Tuned Random Forest

Random Forest hyperparameters were optimized using `RandomizedSearchCV` with cross-validation.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8420 |
| Random Forest | 0.7828 | 0.6118 | 0.4973 | 0.5487 | 0.8219 |
| Tuned Random Forest | 0.8006 | 0.6632 | 0.5053 | 0.5736 | 0.8406 |

### Selected Model

**Logistic Regression** was selected for deployment.

It achieved the highest test ROC-AUC and F1 score among the evaluated configurations.

The trained model is serialized using `joblib` and saved to:

```text
models/logistic_regression.joblib
```

The model artifact is generated during training and is not committed to Git.

**Training implementation:**  
`src/customer_churn_mlops/train.py`

**Model comparison implementation:**  
`src/customer_churn_mlops/compare_models.py`

**Comparison results:**  
[model_comparison.csv](reports/model_comparison.csv)

## System Architecture

The application follows this workflow:

```text
IBM Telco Customer Churn Dataset
                |
                v
       Data Preprocessing
                |
                v
        Model Training
                |
                +------> MLflow Experiment Tracking
                |
                v
      Logistic Regression
                |
                v
       Serialized Pipeline
                |
                v
          FastAPI API
                |
                v
       Pydantic Validation
                |
                v
       Churn Prediction
                |
                +------> Prediction Logging
                |
                +------> Latency Measurement
                |
                v
          JSON Response
```

The application is packaged using Docker and deployed on Render.

GitHub Actions runs automated code quality checks and tests when changes are pushed or pull requests are opened.

## REST API with FastAPI

The application exposes a REST API for customer churn predictions.

The API loads the trained machine learning pipeline and accepts customer information as JSON.

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Check API health |
| POST | `/predict` | Predict customer churn |
| GET | `/docs` | Interactive Swagger documentation |

### Example Prediction Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 1,
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
  "MonthlyCharges": 75.0,
  "TotalCharges": 75.0
}
```

### Example Response Format

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.675
}
```

The response values above are illustrative.

- `churn_prediction`: Binary prediction, where `1` indicates churn and `0` indicates no churn.
- `churn_probability`: Estimated probability of churn, between `0` and `1`.

### API Input Validation

Pydantic validates incoming requests before predictions are generated.

Implemented numerical constraints include:

| Field | Validation |
|---|---|
| `SeniorCitizen` | Integer between 0 and 1 |
| `tenure` | Integer between 0 and 72 |
| `MonthlyCharges` | Non-negative number |
| `TotalCharges` | Non-negative number |

Missing required fields and invalid numerical values return HTTP `422 Unprocessable Entity`.

**API implementation:**  
`src/customer_churn_mlops/api.py`

**Input schemas:**  
`src/customer_churn_mlops/schemas.py`

## Cloud Deployment

The FastAPI application is deployed on **Render** using Docker.

### Deployment Details

| Property | Value |
|---|---|
| Cloud provider | Render |
| Service | `customer-churn-mlops` |
| Deployment method | Docker |
| Application server | Uvicorn |
| Hosting plan | Free |
| Public API | [Swagger UI](https://customer-churn-mlops-89f8.onrender.com/docs) |
| Health check | [Health endpoint](https://customer-churn-mlops-89f8.onrender.com/health) |

The deployed prediction endpoint has been tested successfully with real HTTP requests.

The API returned HTTP `200 OK` and a JSON prediction response.

### Deployment Considerations

The free hosting tier can introduce cold-start delays after periods of inactivity.

The application is intended as a demonstration and portfolio project, not as a production service with guaranteed availability.

## Prediction Logging and Latency Monitoring

The prediction endpoint includes application logging.

For successful predictions, the application records:

- Predicted churn class
- Estimated churn probability
- Prediction processing latency in milliseconds

Latency is measured using Python's `time.perf_counter()`.

Example log format:

```text
Prediction completed: churn_prediction=1, churn_probability=0.7630
Prediction latency: 401.95 ms
```

These example values come from a previously tested deployment request.

Application logs can be inspected through the Render service dashboard.

This provides basic operational visibility.

Advanced model performance monitoring, automated drift detection, and alerting are not yet implemented.

## Experiment Tracking with MLflow

MLflow is integrated into the training workflow.

It records:

- Experiment runs
- Evaluation metrics
- Classifier hyperparameters
- Serialized model artifacts

### Experiment Configuration

| Property | Value |
|---|---|
| Tracking tool | MLflow |
| Experiment name | `customer-churn-prediction` |
| Run name | `logistic-regression-baseline` |
| Model | Logistic Regression |
| Model artifact | `logistic_regression.joblib` |
| Tracking storage | Local MLflow storage |

### Tracked Evaluation Metrics

| Metric | Baseline Result |
|---|---|
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1 Score | 0.6040 |
| ROC-AUC | 0.8420 |

### Hyperparameter Tracking

Classifier hyperparameters are logged using:

```python
mlflow.log_params(
    model.named_steps["classifier"].get_params()
)
```

### Model Artifact Tracking

The serialized model is logged as an MLflow artifact.

```python
mlflow.log_artifact(
    str(MODEL_PATH),
    artifact_path="model"
)
```

### Launch the MLflow Dashboard

Run:

```bash
uv run mlflow ui --host 127.0.0.1 --port 5000
```

Open:

http://localhost:5000

Select the `customer-churn-prediction` experiment to inspect the training runs, evaluation metrics, hyperparameters, and model artifacts.

### Local Tracking Storage

MLflow data is stored locally in:

```text
mlflow.db
mlruns/
```

These files and directories are excluded from Git.

**Implementation:**  
`src/customer_churn_mlops/train.py`

## Getting Started

### Prerequisites

- Python 3.12
- [uv package manager](https://docs.astral.sh/uv/)
- Git

Docker is optional for local containerized execution.

### 1. Clone the Repository

```bash
git clone https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops.git
cd customer-churn-mlops
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Train the Model

```bash
uv run python -m customer_churn_mlops.train
```

This command trains the Logistic Regression pipeline, saves the model artifact, evaluates its performance, and records an MLflow experiment run.

### 4. Start the FastAPI Server

```bash
uv run uvicorn customer_churn_mlops.api:app --reload
```

The application is available at:

- API: http://127.0.0.1:8000
- Swagger documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

## Docker Containerization

The project includes a Dockerfile for running the API in a container.

### Build the Docker Image

First, train the model locally using the training command above.

Then run:

```bash
docker build -t customer-churn-api .
```

### Run the Container

```bash
docker run --rm -p 8000:8000 customer-churn-api
```

Open:

http://localhost:8000/docs

The application runs inside the container using Uvicorn.

## Automated Testing

Automated tests are implemented using pytest.

**Current result: 17 passing tests.**

### Test Coverage

The test suite covers:

- TotalCharges conversion and missing-value handling
- Preservation of original data during preprocessing
- Feature and target separation
- Stratified train/test splitting
- Reproducible dataset splitting
- Preprocessing pipeline structure
- Model pipeline structure
- FastAPI health endpoint
- FastAPI prediction endpoint
- Rejection of missing required fields
- Rejection of negative tenure
- Rejection of tenure above 72 months
- Rejection of negative monthly charges
- Rejection of negative total charges
- Rejection of invalid SeniorCitizen values

### Run All Tests

```bash
uv run pytest tests/ -v
```

Latest verified result:

```text
17 passed, 2 warnings
```

The two warnings originate from third-party dependency deprecations and do not cause test failures.

## Code Quality with Ruff

Ruff is used for Python linting and code formatting.

It is installed as a development dependency.

### Check Code Quality

```bash
uv run ruff check src/ tests/
```

### Check Formatting

```bash
uv run ruff format --check src/ tests/
```

### Automatically Fix Supported Lint Issues

```bash
uv run ruff check src/ tests/ --fix
```

### Format Source Code

```bash
uv run ruff format src/ tests/
```

Ruff checks have been verified locally.

The linting and formatting checks are also integrated into GitHub Actions.

## Continuous Integration with GitHub Actions

The project uses GitHub Actions to automatically validate changes.

The workflow runs when:

- Changes are pushed to `main`
- Pull requests target `main`

### CI Pipeline

The pipeline performs the following operations:

1. Check out the repository
2. Install uv
3. Set up Python 3.12
4. Install project dependencies
5. Check Python code quality with Ruff
6. Check Python formatting with Ruff
7. Train the machine learning model
8. Run the automated test suite

The model is trained during CI so API tests can load the required model artifact without committing binary model files to the repository.

### CI Status

**Latest verified workflow:** CI #10 — Passed

**Commit:** `326c6a4`

**Workflow:**  
[GitHub Actions CI](https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops/actions)

**Configuration:**  
[.github/workflows/ci.yml](.github/workflows/ci.yml)

## Project Structure

```text
customer-churn-mlops/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/
│       └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│   └── logistic_regression.joblib  # Generated locally
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── reports/
│   └── model_comparison.csv
├── src/
│   └── customer_churn_mlops/
│       ├── __init__.py
│       ├── api.py
│       ├── compare_models.py
│       ├── preprocessing.py
│       ├── schemas.py
│       └── train.py
├── tests/
│   ├── test_api.py
│   └── test_preprocessing.py
├── .dockerignore
├── .gitignore
├── .python-version
├── Dockerfile
├── pyproject.toml
├── uv.lock
├── README.md
├── mlflow.db                  # Local, Git-ignored
└── mlruns/                   # Local, Git-ignored
```

## Development Roadmap

- [x] Week 1: Project setup, dataset exploration, and EDA
- [x] Week 2: Data preprocessing and baseline model training
- [x] Week 3: Model comparison and hyperparameter tuning
- [x] Week 4: FastAPI, Docker, automated testing, and GitHub Actions
- [x] Week 5: MLflow experiment tracking
- [x] Week 6: Cloud deployment, prediction logging, and latency measurement
- [x] Week 7: API validation, expanded testing, Ruff, and CI quality checks
- [ ] Week 8: Final portfolio documentation and project presentation

## Future Improvements

Potential future enhancements include:

- MLflow model registry and model version management
- Additional model explainability using SHAP
- Automated data drift detection
- Production model performance monitoring
- Model calibration analysis
- More comprehensive categorical input validation
- Automated deployment workflows
- Improved operational alerting
- Additional security and reliability measures

These are future improvements and are not claimed as completed features.

## Responsible Machine Learning

This project uses historical customer data for educational and portfolio purposes.

Predictions represent estimated probabilities and should not be treated as guaranteed customer outcomes.

Before use in a real business environment, additional work would be required, including:

- Data leakage checks
- Fairness and bias assessment
- Model calibration and reliability evaluation
- Data privacy and security review
- Monitoring for changing customer behavior
- Model performance monitoring
- Operational reliability testing

## License

No license has been specified for this repository.
