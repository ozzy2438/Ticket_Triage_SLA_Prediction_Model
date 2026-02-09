# =============================================================================
# TICKET TRIAGE ML API - DOCKERFILE
# =============================================================================
# 🧒 Çocuğa Anlatım:
# Docker = Taşınabilir kutu
# Uygulamamızı bir kutuya koyuyoruz, her yerde çalışıyor!
# "Bende çalışıyordu" problemi ortadan kalkıyor.
# =============================================================================

# Base image: Python 3.9 slim (küçük boyut)
FROM python:3.9-slim AS base

# Metadata
LABEL maintainer="ML Team"
LABEL description="Ticket Triage + SLA Prediction ML API"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Working directory
WORKDIR /app

# =============================================================================
# BUILD STAGE: Dependencies
# =============================================================================
FROM base AS builder

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# =============================================================================
# PRODUCTION STAGE
# =============================================================================
FROM base AS production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy NLTK data
COPY --from=builder /root/nltk_data /root/nltk_data

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY monitoring/ ./monitoring/

# Create directories
RUN mkdir -p logs checkpoints data

# Non-root user (security best practice)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# DEVELOPMENT STAGE
# =============================================================================
FROM production AS development

# Switch to root for dev tools installation
USER root

# Development tools
RUN pip install \
    pytest \
    pytest-cov \
    black \
    flake8 \
    ipython \
    jupyter

# Copy test files
COPY tests/ ./tests/

# Switch back to appuser
USER appuser

# Development command (with reload)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
