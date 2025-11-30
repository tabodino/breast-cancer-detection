"""
Breast Cancer Detection - Main Streamlit Application
Modular architecture for better maintainability
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Import configurations
from config import setup_page_config, load_custom_css

# Import components
from components.header import render_header
from components.sidebar import render_sidebar

# Import pages (tabs)
from custom_pages.prediction import render_prediction_tab
from custom_pages.metrics import render_metrics_tab
from custom_pages.mlflow_history import render_mlflow_tab
from custom_pages.monitoring import render_monitoring_tab
from custom_pages.history import render_history_tab


def main():
    """Main application entry point."""

    # Setup page
    setup_page_config()
    load_custom_css()

    # Render header
    render_header()

    # Render sidebar and get configuration
    model_path, options = render_sidebar()

    if model_path is None:
        st.warning("⚠️ Please select a model from the sidebar")
        st.stop()

    # Create tabs
    tabs = st.tabs(
        [
            "🔮 Prediction",
            "📊 Model Metrics",
            "📈 MLflow History",
            "🎛️ Monitoring",
            "📝 History",
        ]
    )

    # Render each tab
    with tabs[0]:
        render_prediction_tab(model_path, options)

    with tabs[1]:
        render_metrics_tab()

    with tabs[2]:
        render_mlflow_tab()

    with tabs[3]:
        render_monitoring_tab()

    with tabs[4]:
        render_history_tab()


if __name__ == "__main__":
    main()
