"""Sidebar component for model selection and options."""

import streamlit as st
from pathlib import Path
from src.config import get_settings

settings = get_settings()
MODELS_DIR = Path(settings.models_dir or "models")

# Available models
AVAILABLE_MODELS = {
    "EfficientNet B3": "efficientnet_b3",
    "ResNet50": "resnet50",
    "MobileNet V3": "mobilenet_v3",
    "U-Net": "unet",
    "Custom CNN": "cnn",
}


def render_sidebar():
    """
    Render sidebar with model selection and options.

    Returns:
        tuple: (model_path, options_dict)
    """

    with st.sidebar:
        st.title("⚙️ Configuration")

        # Model Selection
        st.subheader("🤖 Model Selection")

        selected_model_name = st.selectbox(
            "Choose a model",
            options=list(AVAILABLE_MODELS.keys()),
            help="Select the deep learning model for predictions",
            key="model_selector",
        )

        model_key = AVAILABLE_MODELS[selected_model_name]
        model_path = MODELS_DIR / f"best_{model_key}.keras"

        # Model info
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            st.success(f"✅ Model found ({size_mb:.1f} MB)")
        else:
            st.error(f"❌ Model not found: {model_path.name}")
            st.info("Please train the model first or check the path")
            return None, {}

        st.divider()

        # Prediction Options
        st.subheader("🎛️ Prediction Options")

        # Advanced options in expander
        with st.expander("⚙️ Advanced Options"):
            enhance_contrast = st.checkbox(
                "Enhance Contrast",
                value=True,
                help="Apply histogram equalization (CLAHE)",
            )

            show_preprocessing = st.checkbox(
                "Show Preprocessing Steps",
                value=False,
                help="Display before/after preprocessing",
            )

        st.divider()

        # Model Information
        with st.expander("📚 Model Information"):
            st.markdown(f"""
            **Selected Model:** {selected_model_name}
            
            **Architecture Details:**
            - Input Size: 224×224 px
            - Classes: 2 (Benign/Malignant)
            - Framework: TensorFlow/Keras
            
            **Key:** `{model_key}`
            """)

        st.divider()

        # About Section
        st.subheader("ℹ️ About")
        st.markdown("""
        This application uses state-of-the-art deep learning models 
        to detect breast cancer from histopathological images.
        
        **Features:**
        - 5 different CNN architectures
        - Real-time predictions
        - MLflow experiment tracking
        - Comprehensive metrics
        """)

        # Links
        st.markdown("""
        ---
        **Resources:**
        - [GitHub Repository](https://github.com/tabodino/breast-cancer-detection)
        - [Documentation](https://github.com/tabodino/breast-cancer-detection/blob/main/README.md)
        - [MLflow UI](http://localhost:5000)
        """)

    # Return configuration
    options = {
        "model_name": model_key,
        "enhance_contrast": enhance_contrast,
        "show_preprocessing": show_preprocessing,
    }

    return model_path, options
