"""Prediction tab - Main prediction interface."""

import streamlit as st
import numpy as np
from PIL import Image
import cv2
from pathlib import Path

from services.prediction_service import PredictionService
from services.storage_service import save_prediction_history
from utils.image_processing import preprocess_image
from utils.visualization import render_prediction_results


def render_prediction_tab(model_path: Path, options: dict):
    """
    Render the prediction tab.

    Args:
        model_path: Path to the model file
        options: Configuration options from sidebar
    """

    st.header("🔬 Image Analysis")

    # Initialize prediction service
    if "prediction_service" not in st.session_state:
        st.session_state.prediction_service = PredictionService()

    prediction_service = st.session_state.prediction_service

    # Two columns layout
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 Upload Medical Image")

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png", "tiff"],
            help="Upload a histopathological image for analysis",
        )

        if uploaded_file is not None:
            # Load and display image
            image = Image.open(uploaded_file)
            image_array = np.array(image)

            # Ensure RGB
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

            image_array = image_array.astype(np.uint8)

            st.image(image_array, caption="Uploaded Image", width="stretch")

            # Show preprocessing if enabled
            if options.get("show_preprocessing", False):
                st.divider()
                st.subheader("🔍 Preprocessing Steps")

                preprocessed = preprocess_image(
                    image_array, enhance_contrast=options.get("enhance_contrast", True)
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.write("**Original Image**")
                    st.image(image_array, width="stretch")
                    st.caption(f"Size: {image_array.shape[1]}×{image_array.shape[0]}")

                with col_b:
                    st.write("**Preprocessed Image**")
                    # Denormalize for display (preprocessed is normalized to [0,1])
                    display_preprocessed = (preprocessed[0] * 255).astype(np.uint8)
                    st.image(display_preprocessed, width="stretch")
                    st.caption("Resized: 224×224 | Normalized | Enhanced")

                # Show preprocessing steps
                st.info("""
                **Preprocessing Pipeline:**
                1. Convert to RGB (if needed)
                2. Contrast Enhancement (CLAHE) ✓ if enabled
                3. Resize to 224×224
                4. Normalize to [0, 1]
                5. Add batch dimension
                """)

            # Predict button
            st.divider()

            if st.button("🔍 Analyze Image", type="primary", width="stretch"):
                with st.spinner("Loading model..."):
                    # Load model
                    model = prediction_service.load_model(model_path)

                if model is not None:
                    # Set the model in the service
                    prediction_service.model = model

                    with st.spinner("Analyzing image..."):
                        try:
                            # Make prediction
                            class_idx, confidence = prediction_service.predict(
                                image_array
                            )

                            # Save to history
                            save_prediction_history(
                                filename=uploaded_file.name,
                                model_name=options.get("model_name", "unknown"),
                                class_idx=class_idx,
                                confidence=confidence,
                            )

                            # Store in session state
                            st.session_state.current_prediction = {
                                "image": image_array,
                                "class_idx": class_idx,
                                "confidence": confidence,
                                "filename": uploaded_file.name,
                                "model": options.get("model_name"),
                            }

                            st.success("✅ Analysis complete!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
                            st.exception(e)
                else:
                    st.error("Failed to load model")

    with col2:
        # Display results if available
        if "current_prediction" in st.session_state:
            pred = st.session_state.current_prediction

            st.subheader("🎯 Prediction Results")

            render_prediction_results(
                class_idx=pred["class_idx"],
                confidence=pred["confidence"],
                image=pred["image"],
                model_name=pred.get("model", "unknown"),
            )
        else:
            st.info("👈 Upload an image and click 'Analyze' to see results")

            # Show example
            st.markdown("""
            ### How to use:
            1. Upload a histopathological image
            2. Review the image preview
            3. Click 'Analyze Image'
            4. View detailed results here
            
            **Supported formats:** JPG, PNG, TIFF
            """)
