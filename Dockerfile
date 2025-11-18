FROM python:3.12-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e .

COPY . .

# Expose ports
EXPOSE 8000 8501 5000

# Default command
CMD ["bash"]