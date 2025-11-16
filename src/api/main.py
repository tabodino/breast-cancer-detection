from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from PIL import Image
import numpy as np
import tensorflow as tf
from datetime import datetime
import io
import os
import logging
import uvicorn
from src.config import get_settings
from src.utils.mlflow_utils import get_latest_run_id

settings = get_settings()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --- Configuration ---
MODEL_PATH = f"models/model_{get_latest_run_id()}.keras"
IMAGE_SIZE = settings.image_size or (224, 224)

# --- Load model once at startup ---
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    model = tf.keras.models.load_model(str(MODEL_PATH))
    print(f"Model loaded from {MODEL_PATH}")
    yield


# --- Init FastAPI app ---
app = FastAPI(title="Breast Cancer Detector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Helper: Preprocess image ---
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)
    arr = np.array(image).astype(np.float32) / 255.0  # Normalisation
    arr = np.expand_dims(arr, axis=0)  # Batch dimension
    return arr


# --- Predict route ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        arr = preprocess_image(image)
        proba = model.predict(arr)[0][0]  # For sigmoid, batch=1
        label = int(proba > 0.5)
        return {"probability": float(proba), "label": label}
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
