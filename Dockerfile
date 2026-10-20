FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Keep the project layout so config.py resolves models/ from /app
COPY src/ ./src/
COPY api/ ./api/
COPY models/fraud_xgboost_validated_model.joblib ./models/
COPY models/feature_columns.json ./models/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "api.fraud_api:app", "--host", "0.0.0.0", "--port", "8000"]
