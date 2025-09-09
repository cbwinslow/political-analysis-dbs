# Multi-stage Dockerfile for Political Analysis Database System
# Simplified approach without UV in Docker for better compatibility

FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Production stage
FROM base AS production

# Copy requirements first for better caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    requests tqdm psycopg2-binary pydantic fastapi "uvicorn[standard]" \
    python-dotenv sentence-transformers "numpy<2.0.0" scikit-learn \
    beautifulsoup4 lxml PyPDF2 redis neo4j asyncpg aiofiles httpx

# Copy application files
COPY civic_legis_unified.py ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/cache

# Health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8100/health || exit 1' > /app/healthcheck.sh && \
    chmod +x /app/healthcheck.sh

# Expose port
EXPOSE 8100

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Run application with initialization
ENV SKIP_AUTO_INSTALL=1
CMD ["sh", "-c", "python civic_legis_unified.py --init-db && python civic_legis_unified.py --serve --port 8100"]

# Development stage
FROM base AS development
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    requests tqdm psycopg2-binary pydantic fastapi "uvicorn[standard]" \
    python-dotenv sentence-transformers "numpy<2.0.0" scikit-learn \
    beautifulsoup4 lxml PyPDF2 redis neo4j asyncpg aiofiles httpx \
    pytest pytest-asyncio black ruff mypy

COPY . .
ENV SKIP_AUTO_INSTALL=1
CMD ["python", "civic_legis_unified.py", "--serve", "--port", "8100"]