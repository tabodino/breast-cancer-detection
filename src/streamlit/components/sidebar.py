"""
Sidebar component for model selection and configuration.
"""

import streamlit as st
from typing import Optional
from pathlib import Path
from config import AppConfig
from services.prediction_service import PredictionService
from src.config import get_settings

settings = get_settings()
IMAGE_SIZE = settings.image_size or (224, 224)


def render_sidebar() -> Optional[AppConfig]:
    """
    Render sidebar with model selection and configuration.

    Returns:
        AppConfig object or None if configuration is incomplete
    """

    with st.sidebar:
        st.header("⚙️ Configuration")

        interface = PredictionService()
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

        # Advanced options
        with st.expander("Advanced Options"):
            enhance_contrast = st.checkbox("Enhance Contrast", value=True)
            show_preprocessing = st.checkbox("Show Preprocessing Steps", value=False)

        return selected_model, {
            "enhance_contrast": enhance_contrast,
            "show_preprocessing": show_preprocessing,
        }
