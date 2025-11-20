"""
Header component for the application.
"""

import streamlit as st


def render_header():
    """Render application header."""
    st.markdown(
        """
        <div class="header-container">
            <h1>🏥 Breast Cancer Detection AI</h1>
            <p>Advanced Deep Learning Model for Early Cancer Detection</p>
            <p style="font-size: 0.9rem; opacity: 0.9;">
                Powered by MLOps | TensorFlow | Multiple Architectures
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )
