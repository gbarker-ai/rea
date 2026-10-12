# Production Container Image for Google Cloud Run / ADK Agent Server
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY real_estate_agent/ ./real_estate_agent/
COPY tests/ ./tests/
COPY demo.py README.md ./

EXPOSE 8080

# Launch Google ADK API / Web Server bound to Cloud Run $PORT
CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT:-8080} ."]
