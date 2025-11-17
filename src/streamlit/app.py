import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import tensorflow as tf
import json
from datetime import datetime

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


from src.config import get_settings
from src.etl.data_loader import ImagePreprocessor
from src.evaluation.metrics import ModelEvaluator
from src.evaluation.visualization import MetricsVisualizer
from loguru import logger


settings = get_settings()
PROCESSED_DATA_DIR = settings.processed_data_dir or "data/preprocessed"
MODELS_DIR = settings.models_dir or "models"
IMAGE_SIZE = settings.image_size or (224, 224)

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
    <style>
        :root {
            --primary-color: #0066cc;
            --secondary-color: #f0f2f6;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #ffc107;
        }
        
        .header-container {
            background: linear-gradient(135deg, #0066cc 0%, #004499 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #0066cc;
        }
        
        .prediction-container {
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .confidence-bar {
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .confidence-fill-high {
            height: 100%;
            background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.5s ease;
        }
        
        .confidence-fill-low {
            height: 100%;
            background: linear-gradient(90deg, #ffc107, #ff6b6b);
            transition: width 0.5s ease;
        }
    </style>
""",
    unsafe_allow_html=True,
)


class PredictionInterface:
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

            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])

            return class_idx, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise


def render_header():
    """Render page header."""
    st.markdown(
        """
        <div class="header-container">
            <h1>🏥 Breast Cancer Detection AI</h1>
            <p>Advanced Deep Learning Model for Early Cancer Detection</p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                Powered by MLOps | TensorFlow | EfficientNet/ResNet50/U-Net
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render sidebar with configuration."""
    with st.sidebar:
        st.header("⚙️ Configuration")

        interface = PredictionInterface()
        available_models = interface.get_available_models()

        if not available_models:
            st.warning(
                "No trained models found. Please train a model first using the training script."
            )
            return None, None

        selected_model = st.selectbox(
            "Select Model", available_models, format_func=lambda x: Path(x).stem
        )

        st.divider()
        st.subheader("Model Info")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Status", "Ready", "✓")
        with col2:
            st.metric("Input Size", f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}", "pixels")

        st.divider()

        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            0.0,
            1.0,
            0.5,
            help="Minimum confidence for positive prediction",
        )

        # Advanced options
        with st.expander("Advanced Options"):
            enhance_contrast = st.checkbox("Enhance Contrast", value=True)
            show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)

        return selected_model, {
            "confidence_threshold": confidence_threshold,
            "enhance_contrast": enhance_contrast,
            "show_preprocessing": show_preprocessing,
        }


def render_prediction_section(
    interface: PredictionInterface, model_path: str, options: dict
):
    """Render main prediction section."""

    st.header("📋 Image Upload & Analysis")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Upload Image")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a medical image",
            type=["jpg", "jpeg", "png", "tiff"],
            help="Supported formats: JPG, PNG, TIFF",
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            # Ensure RGB
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            st.image(image_array, width=True, caption="Original Image")

            # Load model
            interface.model = interface.load_model(model_path)

            if interface.model is not None:
                # Make prediction
                with st.spinner("Analyzing image..."):
                    try:
                        class_idx, confidence = interface.predict(image_array)

                        # Store results for right column
                        return image_array, class_idx, confidence
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

    return None, None, None


def render_results_section(
    image: np.ndarray, class_idx: int, confidence: float, options: dict
):
    """Render prediction results."""

    st.subheader("Prediction Results")

    class_names = {0: "Benign", 1: "Malignant"}

    # Prediction display
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        prediction_class = class_names.get(class_idx, "Unknown")
        color = "🟢" if class_idx == 0 else "🔴"
        st.metric("Prediction", f"{color} {prediction_class}")

    with col2:
        # Confidence bar
        confidence_pct = confidence * 100
        if confidence > options["confidence_threshold"]:
            status = "✓ Above Threshold"
        else:
            status = "⚠ Below Threshold"

        st.metric("Confidence Score", f"{confidence_pct:.2f}%", status)

    with col3:
        # Risk assessment
        risk_level = "HIGH" if (class_idx == 1 and confidence > 0.7) else "LOW"
        st.metric("Risk Level", risk_level)

    st.divider()

    # Detailed metrics
    st.subheader("Detailed Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model Predictions:**")
        pred_data = {
            "Benign": f"{(1 - confidence) * 100:.2f}%",
            "Malignant": f"{confidence * 100:.2f}%",
        }
        st.table(pred_data)

    with col2:
        st.write("**Image Information:**")
        info_data = {
            "Dimensions": f"{image.shape[1]}×{image.shape[0]}",
            "Channels": f"{image.shape[2] if len(image.shape) > 2 else 1}",
            "Data Type": str(image.dtype),
        }
        st.table(info_data)

    # Preprocessing visualization
    if options.get("show_preprocessing", False):
        st.subheader("Preprocessing Steps")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Image**")
            st.image(image, width=True)

        with col2:
            st.write("**Preprocessed Image**")
            preprocessed = PredictionInterface().preprocessor.preprocess(
                image, enhance_contrast=options.get("enhance_contrast", True)
            )
            # Denormalize for display
            display_preprocessed = ((preprocessed + 1) / 2 * 255).astype(np.uint8)
            st.image(display_preprocessed, width=True)


def render_model_info_section():
    """Render model information section."""
    st.header("📊 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="metric-card">
                <h3>Architecture</h3>
                <p>EfficientNet B3</p>
                <small>Transfer learning with ImageNet weights</small>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="metric-card">
                <h3>Input Size</h3>
                <p>224 × 224 px</p>
                <small>RGB channels with normalization</small>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <h3>Output Classes</h3>
                <p>2 classes</p>
                <small>Benign / Malignant</small>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # MLflow tracking
    with st.expander("View MLflow Experiments"):
        st.info("MLflow UI is available at: `http://localhost:5000`")
        st.code("mlflow ui --backend-store-uri file:./mlruns")


def render_history_section():
    """Render prediction history."""
    st.header("📝 Prediction History")

    history_file = PROCESSED_DATA_DIR / "prediction_history.json"

    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

        st.write(f"Total predictions: {len(history)}")

        # Display recent predictions
        if history:
            recent = sorted(
                history, key=lambda x: x.get("timestamp", ""), reverse=True
            )[:5]

            for pred in recent:
                with st.expander(f"Prediction at {pred.get('timestamp', 'Unknown')}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Class", pred.get("prediction", "Unknown"))
                    with col2:
                        st.metric(
                            "Confidence", f"{pred.get('confidence', 0) * 100:.2f}%"
                        )
                    with col3:
                        st.metric("Risk", pred.get("risk_level", "Unknown"))
    else:
        st.info("No prediction history yet. Upload an image to get started!")


def main():
    """Main application function."""

    # Render header
    render_header()

    # Render sidebar
    model_path, options = render_sidebar()

    if model_path is None:
        st.stop()

    # Create interface
    interface = PredictionInterface()

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Info", "History"])

    with tab1:
        image, class_idx, confidence = render_prediction_section(
            interface, model_path, options
        )

        if image is not None and class_idx is not None:
            render_results_section(image, class_idx, confidence, options)

    with tab2:
        render_model_info_section()

    with tab3:
        render_history_section()


if __name__ == "__main__":
    main()
