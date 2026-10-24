[![CI](https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops/actions/workflows/ci.yml)

# Customer Churn Prediction & MLOps Platform

An end-to-end machine learning engineering portfolio project focused on predicting customer churn, identifying churn-related patterns, and building a reproducible machine learning system using modern software engineering and MLOps practices.

> **Project status:** Exploratory data analysis, data preprocessing, model training and comparison, FastAPI prediction endpoints, Docker containerization, automated testing, GitHub Actions CI, and MLflow experiment tracking completed. Cloud deployment, monitoring, and additional model explainability are planned.

## Project Overview

Customer churn prediction helps businesses identify customers who may discontinue their services.

This project uses the IBM Telco Customer Churn dataset to develop a machine learning solution that predicts whether a customer is likely to churn.

Beyond model development, the project demonstrates practical machine learning engineering skills, including:

- Exploratory data analysis and data quality assessment
- Reproducible preprocessing pipelines
- Supervised machine learning and model evaluation
- Model comparison and hyperparameter tuning
- Model serialization and loading
- REST API development with FastAPI
- Containerization with Docker
- Automated testing with pytest
- Continuous integration using GitHub Actions
- Experiment tracking with MLflow

## Dataset

**IBM Telco Customer Churn Dataset**

The dataset contains:

- 7,043 customer records
- 21 original columns
- Customer demographics
- Subscription and internet services
- Contract and payment information
- Customer tenure and charges
- Customer churn labels

Dataset source: [IBM Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The CSV dataset is stored at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset is included in the repository to support reproducible training in GitHub Actions.

## Exploratory Data Analysis

The initial exploratory analysis investigated customer behavior, churn distribution, and relationships between customer characteristics and churn.

### Key Findings

| Metric | Finding |
| --- | --- |
| Total customers | 7,043 |
| Overall churn rate | 26.54% |
| Month-to-month contract churn | 42.71% |
| One-year contract churn | 11.27% |
| Two-year contract churn | 2.83% |
| Fiber optic customer churn | 41.89% |
| Electronic check customer churn | 45.29% |
| Median tenure of churned customers | 10 months |
| Median tenure of retained customers | 38 months |

### Data Quality

The analysis identified 11 missing `TotalCharges` values, all associated with customers with zero tenure.

These values are handled during preprocessing.

No duplicate customer IDs or fully duplicated rows were identified.

These findings describe associations in the dataset and should not be interpreted as causal relationships.

**EDA notebook:** [01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)

## Data Preprocessing

A reusable preprocessing pipeline was implemented using scikit-learn.

The preprocessing workflow includes:

- Conversion of `TotalCharges` to numeric values
- Handling missing charges for customers with zero tenure
- Separation of input features and target labels
- Removal of customer identifiers from model inputs
- Stratified training and testing split
- Standardization of numerical features
- One-hot encoding of categorical features
- Integration of preprocessing and classification into a single pipeline

The dataset is split into:

- 80% training data
- 20% testing data

A fixed random seed supports reproducible results.

**Implementation:** `src/customer_churn_mlops/preprocessing.py`

## Machine Learning Models

Three model configurations were evaluated:

1. Logistic Regression
2. Random Forest
3. Tuned Random Forest

Random Forest hyperparameters were optimized using `RandomizedSearchCV` with cross-validation.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8420 |
| Random Forest | 0.7828 | 0.6118 | 0.4973 | 0.5487 | 0.8219 |
| Tuned Random Forest | 0.8006 | 0.6632 | 0.5053 | 0.5736 | 0.8406 |

### Selected Model

**Logistic Regression** was selected as the baseline deployment model.

It achieved the highest test ROC-AUC and F1 score among the evaluated configurations.

The trained model is serialized using `joblib` and saved to:

`models/logistic_regression.joblib`

The model artifact is generated locally during training and is not committed to Git.

**Training implementation:** `src/customer_churn_mlops/train.py`

**Model comparison implementation:** `src/customer_churn_mlops/compare_models.py`

**Comparison results:** [model_comparison.csv](reports/model_comparison.csv)

## REST API with FastAPI

A REST API was developed using FastAPI to serve customer churn predictions.

The API loads the trained machine learning pipeline and accepts customer information as JSON.

### Available Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Check API health |
| POST | `/predict` | Predict customer churn |
| GET | `/docs` | Interactive Swagger API documentation |

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

The values above illustrate the response format.

- `churn_prediction`: Binary prediction, where 1 indicates churn and 0 indicates no churn.
- `churn_probability`: Estimated probability of churn.

**API implementation:** `src/customer_churn_mlops/api.py`

**Input validation:** `src/customer_churn_mlops/schemas.py`

## Getting Started

### Prerequisites

- Python 3.12
- [uv package manager](https://docs.astral.sh/uv/)
- Git

Docker is optional for running the containerized application.

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

This creates the trained model artifact required by the API and records an MLflow experiment run.

### 4. Start the FastAPI Server

```bash
uv run uvicorn customer_churn_mlops.api:app --reload
```

The API will be available at:

- API: http://127.0.0.1:8000
- Interactive documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health

## Docker Containerization

The application includes a Dockerfile for containerized execution.

The Docker image installs the project dependencies and includes the trained model artifact.

### Build the Docker Image

First, generate the model locally using the training command above.

Then build the image:

```bash
docker build -t customer-churn-api .
```

### Run the Container

```bash
docker run --rm -p 8000:8000 customer-churn-api
```

Access the API documentation at:

http://localhost:8000/docs

The application runs inside the container using Uvicorn.

## Automated Testing

Automated tests are implemented using pytest.

The tests cover:

- Data cleaning and missing-value handling
- Feature and target separation
- Reproducible train/test splitting
- Preprocessing pipeline structure
- Model pipeline structure
- FastAPI health endpoint
- FastAPI prediction endpoint

### Run All Tests

```bash
uv run pytest tests/ -v
```

**Test suite:** 11 passing tests, verified locally before the MLflow integration.

## Continuous Integration

GitHub Actions provides automated continuous integration for the project.

The CI workflow runs whenever changes are pushed to the `main` branch or a pull request targets `main`.

### CI Pipeline

1. Check out the repository
2. Install uv
3. Install Python 3.12
4. Install project dependencies
5. Train the machine learning model
6. Execute automated tests

Training the model during CI ensures that API tests can load the required model artifact without committing binary model files to Git.

**Workflow configuration:** [.github/workflows/ci.yml](.github/workflows/ci.yml)

**CI status:** Successfully verified on GitHub Actions before the MLflow integration; the updated workflow still needs verification.

## Experiment Tracking with MLflow

MLflow is integrated into the model training workflow to track experiments, evaluation metrics, model hyperparameters, and serialized model artifacts.

Experiment tracking helps maintain reproducibility and provides a structured way to inspect and compare machine learning runs.

### Experiment Configuration

| Property | Value |
| --- | --- |
| Tracking tool | MLflow |
| Experiment name | `customer-churn-prediction` |
| Run name | `logistic-regression-baseline` |
| Model | Logistic Regression |
| Artifact | `logistic_regression.joblib` |
| Tracking storage | Local MLflow storage |

### Tracked Evaluation Metrics

Each training run records the following metrics:

| Metric | Baseline Result |
| --- | --- |
| Accuracy | 0.8055 |
| Precision | 0.6572 |
| Recall | 0.5588 |
| F1 Score | 0.6040 |
| ROC-AUC | 0.8420 |

### Tracked Hyperparameters

The training workflow records Logistic Regression hyperparameters using:

```python
mlflow.log_params(
    model.named_steps["classifier"].get_params()
)
```

This includes model configuration values such as regularization strength, solver, maximum iterations, and other classifier parameters.

### Model Artifact Tracking

The trained scikit-learn pipeline is saved using `joblib` and uploaded to the MLflow run as an artifact.

```python
mlflow.log_artifact(
    str(MODEL_PATH),
    artifact_path="model"
)
```

This allows the serialized pipeline to be inspected and downloaded from the MLflow interface.

### Run a Tracked Experiment

Execute the training script:

```bash
uv run python -m customer_churn_mlops.train
```

The script automatically:

1. Loads and preprocesses the dataset
2. Trains the Logistic Regression pipeline
3. Saves the trained model locally
4. Evaluates model performance
5. Logs evaluation metrics to MLflow
6. Logs classifier hyperparameters
7. Stores the serialized model artifact

### Launch the MLflow Dashboard

Start the MLflow user interface:

```bash
uv run mlflow ui --host 127.0.0.1 --port 5000
```

Open:

http://localhost:5000

Select **Model training**, then open the `customer-churn-prediction` experiment to inspect recorded runs.

The dashboard provides access to:

- Experiment history
- Training run details
- Evaluation metrics
- Model hyperparameters
- Serialized model artifacts

### Local Tracking Storage

MLflow tracking information is stored locally in:

```text
mlflow.db
mlruns/
```

These files and directories are excluded from Git using `.gitignore` because they contain locally generated experiment data.

The MLflow experiment tracking implementation has been tested locally, including successful metric logging and model artifact storage.

**Implementation:** `src/customer_churn_mlops/train.py`

## Technology Stack

### Machine Learning and Data Analysis

- Python 3.12
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

### Backend Development

- FastAPI
- Pydantic
- Uvicorn

### MLOps and Software Engineering

- MLflow
- uv
- joblib
- pytest
- Docker
- Git
- GitHub
- GitHub Actions

### Development Environment

- Ubuntu on WSL2
- Visual Studio Code

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
│   └── logistic_regression.joblib   # Generated locally
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
├── mlflow.db                      # Generated locally, Git-ignored
└── mlruns/                        # Generated locally, Git-ignored
```

## Development Roadmap

- [x] Week 1: Project setup, dataset exploration, and EDA
- [x] Week 2: Data cleaning, preprocessing, and baseline model training
- [x] Week 3: Model comparison, hyperparameter tuning, and model persistence
- [x] REST API development with FastAPI
- [x] Docker containerization
- [x] Automated testing with pytest
- [x] GitHub Actions continuous integration
- [x] Week 5: MLflow experiment tracking, metrics, parameters, and model artifact logging
- [ ] Model registry and version management
- [ ] Cloud deployment
- [ ] Model monitoring and logging
- [ ] Additional model explainability
- [ ] Final portfolio documentation and presentation

The roadmap is iterative and may be adjusted as the project develops.

## Future Improvements

Planned improvements include:

- Model registration and version management using MLflow
- Additional model evaluation and explainability
- Cloud deployment of the FastAPI application
- Prediction logging and monitoring
- Model performance and data drift monitoring
- Improvements to API input validation and error handling
- Expanded automated testing
- Continuous delivery and deployment automation

## Responsible Machine Learning

This project uses historical customer data for educational and portfolio purposes.

Model predictions should not be treated as guaranteed customer outcomes.

Before real-world deployment, additional evaluation would be necessary, including:

- Data leakage checks
- Fairness and bias assessment
- Model calibration and reliability
- Performance monitoring
- Data privacy and security
- Evaluation under changing customer behavior

## License

No license has been specified for this repository.