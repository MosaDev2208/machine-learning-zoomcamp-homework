# Use Python 3.12 slim image
FROM python:3.12-slim-bookworm

# Install uv (The modern dependency manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy the model and prediction script
COPY predict.py model.bin ./

# Expose the port
EXPOSE 8000

# Run the application
ENTRYPOINT ["/app/.venv/bin/uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]