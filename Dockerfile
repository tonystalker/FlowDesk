# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen, no dev tools)
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Final stage — lean runtime image
# ---------------------------------------------------------------------------
FROM python:3.11-slim
WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code (note: .dockerignore controls what gets included)
COPY . .

# Ensure the virtualenv is used by default
ENV PATH="/app/.venv/bin:$PATH"

# Pre-bake the cross-encoder model at build time so it is available in the
# image and does NOT need to be downloaded at container startup (which would
# exceed Cloud Run's startup timeout).
# The model is cached in HuggingFace's default cache dir inside the image.
RUN python -c "\
from sentence_transformers import CrossEncoder; \
print('Pre-loading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Cross-encoder model cached successfully.')"

# Cloud Run injects PORT automatically; default to 8080 as fallback.
# Alembic migrations run best-effort — failure is non-fatal so uvicorn
# always starts and the container binds to the port within the timeout.
CMD ["sh", "-c", "\
  echo 'Starting FlowDesk support-gateway...' && \
  (alembic upgrade head && echo 'Migrations applied.') || echo 'Alembic skipped (non-fatal).' && \
  exec uvicorn gateway.main:app --host 0.0.0.0 --port ${PORT:-8080} --log-level info"]
