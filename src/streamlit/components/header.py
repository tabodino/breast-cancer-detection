"""Header component for the application."""

import streamlit as st


def render_header():
    """Render the main application header."""

    st.markdown(
        """
        <div class="header-container">
            <h1>🏥 Breast Cancer Detection AI</h1>
            <p style="font-size: 1.2rem; margin-top: 0.5rem;">
                Deep Learning-Powered Medical Image Analysis
            </p>
            <p style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
                Multi-Model Architecture | MLflow Tracking | Production Ready
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )
