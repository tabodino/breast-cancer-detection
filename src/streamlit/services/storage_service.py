"""Storage service for prediction history."""

import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from src.config import get_settings

settings = get_settings()
DATA_DIR = Path(settings.processed_data_dir or "data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_prediction_history(
    filename: str, model_name: str, class_idx: int, confidence: float
):
    """
    Save prediction to history file.

    Args:
        filename: Name of the uploaded file
        model_name: Name of the model used
        class_idx: Predicted class index
        confidence: Prediction confidence
    """
    history_file = DATA_DIR / "prediction_history.json"

    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
            history = []

    # Prepare new entry
    class_names = {0: "Benign", 1: "Malignant"}
    prob_malignant = confidence if class_idx == 1 else (1 - confidence)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "model": model_name,
        "prediction": class_names.get(class_idx, "Unknown"),
        "confidence": float(confidence),
        "risk_level": "HIGH" if prob_malignant > 0.7 else "LOW",
    }

    # Add to history
    history.append(entry)

    # Keep only last 1000 predictions
    history = history[-1000:]

    # Save
    try:
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Saved prediction to history: {filename}")
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")


def load_prediction_history():
    """
    Load prediction history from file.

    Returns:
        list: List of prediction entries
    """
    history_file = DATA_DIR / "prediction_history.json"

    if not history_file.exists():
        return []

    try:
        with open(history_file, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load history: {e}")
        return []


def clear_prediction_history():
    """Clear all prediction history."""
    history_file = DATA_DIR / "prediction_history.json"

    try:
        with open(history_file, "w") as f:
            json.dump([], f)
        logger.info("Prediction history cleared")
        return True
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        return False
