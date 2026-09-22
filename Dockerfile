# AskingMe Agent enterprise policy Q&A system — Docker multi-stage build
# Goal: keep the production image minimal while including debugging tools in development

# ── Stage 1: base environment ─────────────────────────────────────────────────
FROM python:3.12-slim AS base

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# curl is used by health checks; gcc/g++ are no longer needed after removing local ML models
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: install Python dependencies ──────────────────────────────────────
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download ChromaDB's bundled ONNX embedding model (~79 MB) to avoid runtime timeouts
RUN mkdir -p /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
    curl -L --retry 3 --retry-delay 5 -o /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz \
    https://chroma-onnx-models.s3.amazonaws.com/all-MiniLM-L6-v2/onnx.tar.gz && \
    cd /root/.cache/chroma/onnx_models/all-MiniLM-L6-v2 && \
    tar -xzf onnx.tar.gz && \
    rm onnx.tar.gz

# ── Stage 3: production image ─────────────────────────────────────────────────
FROM base AS production

# Run as a non-root user. Create it first so COPY can set ownership without an extra chown layer.
RUN useradd -m -u 1000 askingme

# Copy installed packages from the dependency stage
COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin
# Copy the pre-downloaded ONNX model cache
COPY --from=dependencies --chown=askingme:askingme /root/.cache/chroma /home/askingme/.cache/chroma

# Copy application code
COPY --chown=askingme:askingme . .

# Create required directories and change ownership only where runtime writes are needed.
RUN mkdir -p /app/data/chroma /app/logs /app/config && \
    chown askingme:askingme /app/data /app/data/chroma /app/logs /app/config
USER askingme

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ── Stage 4: development image ────────────────────────────────────────────────
FROM dependencies AS development

COPY . .

RUN mkdir -p /app/data/chroma /app/logs /app/config /app/tests && \
    chmod -R 777 /app/data /app/logs

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
