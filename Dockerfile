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
RUN python -c "\
from sentence_transformers import CrossEncoder; \
print('Pre-loading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Cross-encoder model cached successfully.')"

# Cloud Run injects PORT automatically; default to 8080 as fallback.
# Alembic migrations run best-effort — failure is non-fatal so uvicorn
# always starts and the container binds to the port within the timeout.
# Using a startup script to guarantee uvicorn always runs regardless of
# alembic exit code (avoiding shell precedence issues with && and ||).
COPY <<'EOF' /app/start.sh
#!/bin/sh
set -e
echo "Starting FlowDesk support-gateway..."
alembic upgrade head 2>&1 && echo "Migrations applied." || echo "Alembic skipped (non-fatal)."
echo "Launching uvicorn on port ${PORT:-8080}..."
exec uvicorn gateway.main:app --host 0.0.0.0 --port "${PORT:-8080}" --log-level info
EOF
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
