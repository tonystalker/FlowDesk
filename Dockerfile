# syntax=docker/dockerfile:1
FROM python:3.11-slim AS builder
WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen, no dev tools)
RUN uv sync --frozen --no-dev

# Cloud Run has no GPU — swap CUDA torch for CPU-only and strip ~2.5 GB of
# NVIDIA libraries that sentence-transformers pulls in transitively.
RUN .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
        --force-reinstall --no-deps && \
    .venv/bin/pip uninstall -y \
        nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
        nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver \
        nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
        nvidia-nvshmem-cu13 nvidia-nvtx triton cuda-bindings cuda-pathfinder \
        cuda-toolkit 2>/dev/null || true

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

# Make the startup script executable
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
