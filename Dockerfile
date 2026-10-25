# ==========================================
# STAGE 1: Builder & Model Training Stage
# ==========
FROM python:3.10-slim AS builder

WORKDIR /app

# Dependency layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy dataset and training script
COPY data/ /app/data/
COPY training/ /app/training/
COPY params.yaml /app/params.yaml

# Train model inside container
RUN python training/train.py --data-path data/raw/breast_cancer.csv --output-dir models --model-version v1.0

# ==========================================
# STAGE 2: Production Serving Stage
# ==========================================
FROM python:3.10-slim AS runner

# Build Arguments
ARG MODEL_VERSION="v1.0"
ARG GIT_SHA="unknown"

# Environment Variables & Labels
ENV MODEL_VERSION=${MODEL_VERSION} \
    GIT_SHA=${GIT_SHA} \
    LOG_LEVEL="INFO" \
    MODEL_PATH="models/model.joblib" \
    DECISION_THRESHOLD="0.5" \
    PYTHONUNBUFFERED=1

LABEL org.opencontainers.image.version="${MODEL_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}"

# Install curl for HEALTHCHECK and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 appuser

WORKDIR /app

# Install runtime dependencies first (Cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy trained model artifact from builder stage (NO dataset or training scripts)
COPY --from=builder /app/models/model.joblib /app/models/model.joblib

# Copy FastAPI application code
COPY app /app/app

# Set non-root user permissions
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health Check Definition
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Exec-form ENTRYPOINT & CMD
ENTRYPOINT ["python", "-m", "uvicorn"]
CMD ["app.main:app", "--host", "0.0.0.0", "--port", "8000"]