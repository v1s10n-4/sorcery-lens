# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Venv — no --user installs, no PATH ambiguity
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download CLIP ViT-B/32 weights into a known cache dir.
# Without this, the first request triggers a 354MB download at runtime
# inside an async handler → blocks event loop → anyio TaskGroup crash.
ENV TORCH_HOME=/opt/torch_cache
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: non-root user
RUN groupadd -r lens && useradd -r -g lens -d /app -s /bin/false lens

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv — fully self-contained
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded model weights — zero network dependency at runtime
COPY --from=builder /opt/torch_cache /opt/torch_cache
ENV TORCH_HOME=/opt/torch_cache

# Copy application code
COPY app/ ./app/

# Data directory — embeddings mounted at runtime via Render disk/env
RUN mkdir -p data && chown lens:lens data

USER lens

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# python -m uvicorn is robust regardless of PATH ordering
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
