"""Visualization utilities for results display."""

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render_prediction_results(
    class_idx: int, confidence: float, image: np.ndarray, model_name: str
):
    """
    Render prediction results with visualizations.

    Args:
        class_idx: Predicted class index (0=Benign, 1=Malignant)
        confidence: Prediction confidence [0, 1]
        image: Original image
        model_name: Name of the model used
    """
    class_names = {0: "Benign", 1: "Malignant"}
    prediction_class = class_names.get(class_idx, "Unknown")

    # Main prediction display
    if class_idx == 0:
        st.success(f"### 🟢 {prediction_class}")
    else:
        st.error(f"### 🔴 {prediction_class}")

    st.divider()

    # Metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Confidence", f"{confidence * 100:.2f}%")

    with col2:
        prob_malignant = confidence if class_idx == 1 else (1 - confidence)
        risk = "HIGH ⚠️" if prob_malignant > 0.7 else "LOW ✓"
        st.metric("Risk Level", risk)

    with col3:
        st.metric("Model Used", model_name.upper())

    st.divider()

    # Probability breakdown
    st.subheader("📊 Probability Distribution")

    prob_benign = confidence if class_idx == 0 else (1 - confidence)
    prob_malignant = confidence if class_idx == 1 else (1 - confidence)

    # Bar chart
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Benign", "Malignant"],
                y=[prob_benign * 100, prob_malignant * 100],
                marker_color=["#28a745", "#dc3545"],
                text=[f"{prob_benign * 100:.1f}%", f"{prob_malignant * 100:.1f}%"],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        yaxis_title="Probability (%)",
        height=300,
        showlegend=False,
        yaxis_range=[0, 100],
    )

    st.plotly_chart(fig, width="stretch")

    # Confidence gauge
    st.subheader("🎯 Confidence Level")

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=confidence * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Confidence Score"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 50], "color": "lightgray"},
                    {"range": [50, 75], "color": "gray"},
                    {"range": [75, 100], "color": "darkgray"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 90,
                },
            },
        )
    )

    fig.update_layout(height=300)
    st.plotly_chart(fig, width="stretch")

    # Image information
    st.subheader("🔍 Image Details")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Dimensions:** {image.shape[1]}×{image.shape[0]} pixels")
        st.write(f"**Channels:** {image.shape[2] if len(image.shape) > 2 else 1}")

    with col2:
        st.write(f"**Data Type:** {image.dtype}")
        st.write(f"**Size:** {image.nbytes / 1024:.1f} KB")

    # Recommendation
    st.divider()

    if class_idx == 1 and prob_malignant > 0.7:
        st.warning("""
        ⚠️ **High Risk Detected**
        
        This image shows characteristics associated with malignancy. 
        We strongly recommend:
        - Immediate consultation with a medical professional
        - Additional diagnostic tests
        - Biopsy if recommended by physician
        
        **Note:** This is an AI-assisted tool and should not replace professional medical diagnosis.
        """)
    elif class_idx == 1:
        st.info("""
        ℹ️ **Moderate Risk**
        
        The model detected potential malignancy indicators. 
        Please consult with a healthcare provider for further evaluation.
        """)
    else:
        st.success("""
        ✅ **Low Risk**
        
        The image appears to show benign characteristics. 
        However, regular screening and professional consultation are still recommended.
        """)
