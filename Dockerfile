FROM python:3.12-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/

RUN uv sync --frozen --no-dev

COPY data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv ./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

RUN uv run python -m customer_churn_mlops.train

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "customer_churn_mlops.api:app", "--host", "0.0.0.0", "--port", "8000"]