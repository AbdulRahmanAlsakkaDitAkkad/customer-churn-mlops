# Customer Churn Prediction & MLOps Platform

An end-to-end machine learning engineering portfolio project focused on predicting customer churn, explaining churn drivers, and building a reproducible ML system with deployment and monitoring capabilities.

> **Project status:** Week 1 completed — project setup and exploratory data analysis. Model training, API development, deployment, and monitoring are planned.

## Project Objectives

- Explore customer behavior and identify patterns associated with churn.
- Build reproducible data preprocessing and feature engineering pipelines.
- Train and evaluate classification models for churn prediction.
- Expose predictions through a REST API.
- Introduce MLOps practices, including experiment tracking, automated testing, CI/CD, containerization, and monitoring.

## Dataset

**IBM Telco Customer Churn**, available on Kaggle.

The dataset contains 7,043 customer records and 21 columns, including customer demographics, subscription services, contract information, payment methods, and churn labels.

The raw dataset is excluded from version control. To reproduce the analysis, download the dataset from Kaggle and place `WA_Fn-UseC_-Telco-Customer-Churn.csv` inside `data/raw/`.

## Tech Stack

**Currently used**
- Python 3.12
- uv for dependency and environment management
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- Jupyter Notebook
- Git and GitHub
- Ubuntu on WSL2

**Planned**
- FastAPI for model inference
- Docker for containerization
- Automated testing and CI/CD
- Experiment tracking and model monitoring

## Week 1 — Exploratory Data Analysis

Completed an initial exploratory analysis of the IBM Telco Customer Churn dataset.

### Key Findings

| Metric | Finding |
| --- | --- |
| Customer records | 7,043 |
| Overall churn rate | 26.54% |
| Month-to-month contract churn | 42.71% |
| One-year contract churn | 11.27% |
| Two-year contract churn | 2.83% |
| Fiber optic customer churn | 41.89% |
| Electronic check customer churn | 45.29% |
| Median tenure of churned customers | 10 months |
| Median tenure of retained customers | 38 months |

Additional data quality checks identified 11 blank `TotalCharges` values, all associated with customers whose tenure was zero. No duplicate customer IDs or fully duplicated rows were found.

These findings describe associations in the dataset and do not establish causation.

**EDA notebook:** [01_exploratory_data_analysis.ipynb](notebooks/01_exploratory_data_analysis.ipynb)

## Getting Started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

### Setup

```bash
git clone https://github.com/AbdulRahmanAlsakkaDitAkkad/customer-churn-mlops.git
cd customer-churn-mlops
uv sync
```

Download the IBM Telco Customer Churn dataset from Kaggle and save the CSV file at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Open the notebook in VS Code or another Jupyter-compatible environment.

## Development Roadmap

- [x] Week 1: Project setup, dataset exploration, and EDA
- [ ] Week 2: Data cleaning, preprocessing, and baseline model training
- [ ] Week 3: Model comparison, evaluation, and explainability
- [ ] Week 4: Reproducible training pipeline and experiment tracking
- [ ] Week 5: REST API for churn predictions
- [ ] Week 6: Docker and automated testing
- [ ] Week 7: CI/CD and deployment
- [ ] Week 8: Monitoring, documentation, and portfolio polish

The roadmap is a working plan and may be adjusted as the project develops.

## Current Project Structure

```text
customer-churn-mlops/
├── data/
│   └── raw/                 # Local dataset (Git-ignored)
├── notebooks/
│   └── 01_exploratory_data_analysis.ipynb
├── src/
│   └── customer_churn_mlops/
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
└── README.md
```

## Responsible Interpretation

This project uses historical customer data for educational and portfolio purposes. Churn predictions should be evaluated for data leakage, class imbalance, model reliability, and potentially unfair outcomes before any real-world use.