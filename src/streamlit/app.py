import streamlit as st
import numpy as np
import cv2
from PIL import Image
import json
import sys

from config import setup_page_config, load_custom_css
from components.header import render_header
from components.sidebar import render_sidebar
from services.prediction_service import PredictionService
from src.config import get_settings
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


settings = get_settings()
PROCESSED_DATA_DIR = settings.processed_data_dir or "data/preprocessed"
MODELS_DIR = settings.models_dir or "models"

setup_page_config()
load_custom_css()


def render_prediction_section(
    interface: PredictionService, model_path: str, options: dict
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

            image_array = image_array.astype(np.uint8)
            st.image(image_array, use_container_width=True, caption="Original Image")

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
        status = f"Confidence: {confidence_pct:.2f}%"
        st.metric("Confidence Score", f"{confidence_pct:.2f}%", status)
        st.caption(status)

    with col3:
        prob_malignant = confidence if class_idx == 1 else (1 - confidence)
        # Risk assessment
        risk_level = "HIGH" if prob_malignant > 0.7 else "LOW"
        # risk_level = "HIGH" if (class_idx == 1 and confidence > 0.7) else "LOW"
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
            st.image(image, use_container_width=True)

        with col2:
            st.write("**Preprocessed Image**")
            preprocessed = PredictionService().preprocessor.preprocess(
                image, enhance_contrast=options.get("enhance_contrast", True)
            )
            # Denormalize for display
            display_preprocessed = ((preprocessed + 1) / 2 * 255).astype(np.uint8)
            st.image(display_preprocessed, use_container_width=True)


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
    interface = PredictionService()

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
