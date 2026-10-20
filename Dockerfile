FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TSHEPO_LOG_LEVEL=INFO \
    TSHEPO_JSON_LOGS=1

# System deps: minimal; no build essentials needed for pure-python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching).
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e ".[api]"

# Copy application code.
COPY tshepo/ ./tshepo/
COPY data/ ./data/
COPY models/ ./models/

# Non-root runtime user.
RUN useradd -m -u 10001 tshepo && chown -R tshepo:tshepo /app
USER tshepo

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "from tshepo.api import app; from fastapi.testclient import TestClient; assert TestClient(app).get('/health').status_code == 200" || exit 1

ENTRYPOINT ["python", "-m", "tshepo.cli.serve", "--host", "0.0.0.0", "--port", "8080"]
