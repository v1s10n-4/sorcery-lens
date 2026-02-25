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

# Pre-download CLIP ViT-B/32 weights.
# open_clip's openai loader is hardcoded to os.path.expanduser("~/.cache/clip")
# — it ignores XDG_CACHE_HOME and TORCH_HOME. In the build stage root's home
# is /root, so weights land at /root/.cache/clip. We copy that into the
# runtime image below and place it where the lens user's ~/ resolves to.
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: non-root user. Home set to /home/lens (writable, not /app).
RUN groupadd -r lens && useradd -r -g lens -d /home/lens -s /bin/false -m lens

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv — fully self-contained
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy pre-downloaded weights into the lens user's ~/.cache/clip so open_clip
# finds them without any network access at runtime.
# Must run before USER lens (we're still root here, so we can chown).
RUN mkdir -p /home/lens/.cache
COPY --from=builder /root/.cache/clip /home/lens/.cache/clip
RUN chown -R lens:lens /home/lens/.cache

# Copy application code
COPY app/ ./app/

# Data directory — embeddings mounted at runtime via Render disk/env
RUN mkdir -p data && chown lens:lens data

USER lens

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOME=/home/lens

EXPOSE 8000

# python -m uvicorn is robust regardless of PATH ordering
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
