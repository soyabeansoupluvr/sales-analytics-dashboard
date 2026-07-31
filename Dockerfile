# Interactive Sales Analytics Dashboard — production image
#
# Build:  docker build -t sales-analytics-dashboard:v1.0.0 .
# Run:    docker compose up   (see docker-compose.yml)

FROM python:3.12-slim

# PYTHONPATH makes the src package importable, matching how
# `python -m streamlit` behaves when run from the repo root.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install pinned dependencies first so layer caching survives code edits.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code only. Data, logs, and configuration enter at runtime.
COPY src/ src/

# Run as a non-root user. Mounted data and log directories are owned by it.
RUN useradd --create-home --uid 1000 dashboard \
    && mkdir -p data/raw data/processed logs \
    && chown -R dashboard:dashboard /app
USER dashboard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "src/app.py"]
