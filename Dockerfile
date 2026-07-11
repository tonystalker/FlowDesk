# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen)
RUN uv sync --frozen --no-dev

# Final stage
FROM python:3.11-slim
WORKDIR /app

# Copy the virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Ensure the virtualenv is used by default
ENV PATH="/app/.venv/bin:$PATH"

# Run FastAPI
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8080"]
