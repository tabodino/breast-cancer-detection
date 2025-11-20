#!/bin/bash
# Script to launch FastAPI, Streamlit, and MLflow in parallel

# ----- Configurable variables -----
FASTAPI_MODULE="src.api.main:app"
STREAMLIT_SCRIPT="src/streamlit/app.py"
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
MLFLOW_PORT=5000

# Emplacement du dossier où sont stockés les logs/artifacts MLflow (adapte selon ton projet)
MLFLOW_ARTIFACT_ROOT="mlruns"

# Set PYTHONPATH to repo root for src/
export PYTHONPATH=$(pwd)

# ---- Start FastAPI ----
echo "Starting FastAPI on port $FASTAPI_PORT..."
uv run uvicorn "$FASTAPI_MODULE" --host 0.0.0.0 --port $FASTAPI_PORT &
FASTAPI_PID=$!

# ---- Start Streamlit ----
echo "Starting Streamlit on port $STREAMLIT_PORT..."
uv run streamlit run "$STREAMLIT_SCRIPT" --server.port=$STREAMLIT_PORT &
STREAMLIT_PID=$!

# ---- Start MLflow Tracking Server ----
echo "Starting MLflow Tracking Server on port $MLFLOW_PORT..."
# Si MLflow n'est pas installé par uv, adapte la commande
uv run mlflow server --backend-store-uri "$MLFLOW_ARTIFACT_ROOT" --default-artifact-root "$MLFLOW_ARTIFACT_ROOT" --host 0.0.0.0 --port $MLFLOW_PORT &
MLFLOW_PID=$!

echo
echo "FastAPI PID:     $FASTAPI_PID  (http://localhost:$FASTAPI_PORT)"
echo "Streamlit PID:   $STREAMLIT_PID  (http://localhost:$STREAMLIT_PORT)"
echo "MLflow PID:      $MLFLOW_PID  (http://localhost:$MLFLOW_PORT)"
echo "Stop all: Ctrl+C"
echo

# ---- Wait all background processes (Ctrl+C stops all) ----
wait
