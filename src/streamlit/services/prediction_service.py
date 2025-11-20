import streamlit as st
from src.etl.data_loader import ImagePreprocessor
from loguru import logger
import numpy as np
import tensorflow as tf
from src.config import get_settings


settings = get_settings()
MODELS_DIR = settings.models_dir or "models"


class PredictionService:
    """Handles prediction interface logic."""

    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.model = None
        self.model_path = None

    @st.cache_resource
    def load_model(_self, model_path: str):
        """Load model with caching."""
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"Error loading model: {e}")
            return None

    def get_available_models(self) -> list:
        """Get list of available trained models."""
        if not MODELS_DIR.exists():
            return []

        models = list(MODELS_DIR.glob("*.keras"))
        return [str(m) for m in models]

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for prediction."""
        try:
            processed = self.preprocessor.preprocess(image, enhance_contrast=True)
            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise

    def predict(self, image: np.ndarray) -> tuple:
        """Make prediction on image."""
        if self.model is None:
            st.error("No model loaded. Please select a model first.")
            return None, None

        try:
            preprocessed = self.preprocess_image(image)
            predictions = self.model.predict(preprocessed, verbose=0)

            if predictions.shape[-1] == 1:
                prob_malignant = predictions[0][0]

                if prob_malignant >= 0.5:
                    class_idx = 1  # Malignant
                    confidence = float(prob_malignant)
                else:
                    class_idx = 0  # Benign
                    confidence = float(1.0 - prob_malignant)

            elif predictions.shape[-1] == 2:
                class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][class_idx])

            else:
                logger.error(f"Unexpected prediction output shape: {predictions.shape}")
                class_idx = 0
                confidence = 0.0

            return class_idx, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
