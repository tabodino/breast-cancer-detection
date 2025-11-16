#!/bin/bash

set -e

echo "Deploying Breast Cancer Detection Application..."

# Build Docker image
echo "Building Docker image..."
docker build -t breast-cancer-detection:latest -f Dockerfile.prod .

# Start services with docker-compose
echo "Starting services..."
docker-compose up -d

echo ""
echo "Deployment complete!"
echo ""
echo "Services available at:"
echo "  - MLflow UI: http://localhost:5000"
echo "  - FastAPI: http://localhost:8000"
echo "  - Streamlit: http://localhost:8501"
echo ""
echo "Health check:"
docker-compose exec api curl http://localhost:8000/health

echo "Done!"
