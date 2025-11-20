"""
Streamlit application configuration and styling.
"""

import streamlit as st
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration."""

    model_type: str
    model_path: str
    confidence_threshold: float
    enhance_contrast: bool
    show_preprocessing: bool


def setup_page_config():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="Breast Cancer Detection AI",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def load_custom_css():
    """Load custom CSS styling."""
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
            
            .benign-badge {
                background: #28a745;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
            }
            
            .malignant-badge {
                background: #dc3545;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
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


# Model architecture mapping
MODEL_ARCHITECTURES = {
    "cnn": {
        "name": "CNN",
        "description": "Simple CNN architecture from scratch",
        "input_size": (224, 224),
    },
    "efficientnet_b3": {
        "name": "EfficientNet B3",
        "description": "Transfer learning with ImageNet weights",
        "input_size": (224, 224),
    },
    "resnet50": {
        "name": "ResNet50",
        "description": "Deep residual network architecture",
        "input_size": (224, 224),
    },
    "mobilenet_v3": {
        "name": "MobileNet V3",
        "description": "Lightweight mobile-optimized network",
        "input_size": (224, 224),
    },
    "unet": {
        "name": "U-Net",
        "description": "Medical image segmentation architecture",
        "input_size": (224, 224),
    },
}

# Class information
CLASS_INFO = {
    0: {
        "name": "Benign",
        "emoji": "🟢",
        "color": "#28a745",
        "description": "Non-cancerous tissue",
    },
    1: {
        "name": "Malignant",
        "emoji": "🔴",
        "color": "#dc3545",
        "description": "Cancerous tissue detected",
    },
}
