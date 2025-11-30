"""Prediction service for model loading and inference."""

import streamlit as st
import numpy as np
from pathlib import Path
from tensorflow import keras
from loguru import logger

from utils.image_processing import preprocess_image


class PredictionService:
    """Service for handling model predictions."""

    def __init__(self):
        """Initialize prediction service."""
        self.model = None
        self.model_path = None

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model_cached(model_path: Path):
        """
        Load a Keras model with caching (static method for proper caching).

        Args:
            model_path: Path to the model file

        Returns:
            Loaded Keras model or None
        """
        try:
            logger.info(f"Loading model from {model_path}")

            # Resolve symlinks if any
            actual_path = model_path.resolve()

            if not actual_path.exists():
                logger.error(f"Model file not found: {actual_path}")
                return None

            model = keras.models.load_model(actual_path)
            logger.success(f"Model loaded successfully: {model_path.name}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            st.error(f"Model loading error: {e}")
            return None

    def load_model(self, model_path: Path):
        """
        Load a Keras model and store it in the service.

        Args:
            model_path: Path to the model file

        Returns:
            Loaded Keras model or None
        """
        self.model_path = model_path
        self.model = self._load_model_cached(model_path)
        return self.model

    def predict(self, image: np.ndarray, enhance_contrast: bool = True):
        """
        Make prediction on an image.

        Args:
            image: Input image as numpy array
            enhance_contrast: Whether to enhance contrast

        Returns:
            tuple: (class_index, confidence)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Preprocess image
        preprocessed = preprocess_image(image, enhance_contrast=enhance_contrast)

        # Make prediction
        prediction = self.model.predict(preprocessed, verbose=0)

        # Handle binary classification
        if prediction.shape[-1] == 1:
            # Sigmoid output
            confidence = float(prediction[0][0])
            class_idx = 1 if confidence > 0.5 else 0
            confidence = confidence if class_idx == 1 else (1 - confidence)
        else:
            # Softmax output
            class_idx = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][class_idx])

        logger.info(f"Prediction: class={class_idx}, confidence={confidence:.3f}")

        return class_idx, confidence
