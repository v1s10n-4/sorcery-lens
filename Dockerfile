# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for CLIP + FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: non-root user
RUN groupadd -r lens && useradd -r -g lens -d /app -s /bin/false lens

WORKDIR /app

# Runtime deps only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /home/lens/.local

# Copy application code
COPY app/ ./app/

# Data directory (embeddings mounted at runtime via bind mount or volume)
RUN mkdir -p data && chown lens:lens data

USER lens

ENV PATH="/home/lens/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
