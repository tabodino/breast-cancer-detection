FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

COPY . .

# Install dependencies
RUN uv pip install --system -e .

# Expose ports
EXPOSE 8000 8501 5000

# Default command
CMD ["bash", "-c", "mlflow ui --backend-store-uri file:./mlruns & uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/streamlit/app.py"]