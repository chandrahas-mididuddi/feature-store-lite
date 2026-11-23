FROM python:3.11-slim

WORKDIR /app

# pyarrow wheels need no extra system libs on slim, but keep the image lean.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_store/ ./feature_store/
COPY scripts/ ./scripts/

ENV FEATURE_STORE_DATA_DIR=/app/data
EXPOSE 8000

CMD ["uvicorn", "feature_store.service:app", "--host", "0.0.0.0", "--port", "8000"]
