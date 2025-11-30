"""History tab - Prediction history and audit trail."""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

from src.config import get_settings

settings = get_settings()
DATA_DIR = Path(settings.processed_data_dir or "data/processed")


def render_history_tab():
    """Render the prediction history tab."""

    st.header("📝 Prediction History")

    history_file = DATA_DIR / "prediction_history.json"

    if not history_file.exists():
        st.info("""
        ### No prediction history yet
        
        Make your first prediction in the **Prediction** tab to see history here.
        
        **What's tracked:**
        - Timestamp
        - Filename
        - Model used
        - Prediction result
        - Confidence score
        - Risk level
        """)
        return

    # Load history
    with open(history_file, "r") as f:
        history = json.load(f)

    if not history:
        st.info("History file exists but is empty. Make a prediction!")
        return

    # Summary
    st.subheader("📊 Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Predictions", len(history))

    with col2:
        benign_count = sum(1 for h in history if h.get("prediction") == "Benign")
        st.metric("Benign", benign_count)

    with col3:
        malignant_count = sum(1 for h in history if h.get("prediction") == "Malignant")
        st.metric("Malignant", malignant_count)

    with col4:
        high_risk = sum(1 for h in history if h.get("risk_level") == "HIGH")
        st.metric("High Risk", high_risk)

    st.divider()

    # Filters
    st.subheader("🔍 Filter History")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Prediction filter
        predictions = list(set(h.get("prediction", "Unknown") for h in history))
        pred_filter = st.multiselect(
            "Prediction", options=predictions, default=predictions
        )

    with col2:
        # Model filter
        models = list(set(h.get("model", "unknown") for h in history))
        model_filter = st.multiselect("Model", options=models, default=models)

    with col3:
        # Date range
        show_recent = st.selectbox(
            "Show", options=["All", "Last 10", "Last 20", "Last 50"], index=1
        )

    # Apply filters
    filtered_history = [
        h
        for h in history
        if h.get("prediction") in pred_filter and h.get("model") in model_filter
    ]

    # Sort by timestamp (most recent first)
    filtered_history = sorted(
        filtered_history, key=lambda x: x.get("timestamp", ""), reverse=True
    )

    # Limit results
    if show_recent != "All":
        limit = int(show_recent.split()[-1])
        filtered_history = filtered_history[:limit]

    st.write(f"**Showing {len(filtered_history)} predictions**")

    st.divider()

    # Display options
    view_mode = st.radio(
        "View Mode", options=["Detailed Cards", "Table View"], horizontal=True
    )

    if view_mode == "Detailed Cards":
        # Card view
        for i, pred in enumerate(filtered_history):
            with st.expander(
                f"#{len(history) - i} - {pred.get('filename', 'Unknown')} - {pred.get('timestamp', 'Unknown')[:19]}"
            ):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    prediction = pred.get("prediction", "Unknown")
                    icon = "🟢" if prediction == "Benign" else "🔴"
                    st.metric("Prediction", f"{icon} {prediction}")

                with col2:
                    confidence = pred.get("confidence", 0) * 100
                    st.metric("Confidence", f"{confidence:.2f}%")

                with col3:
                    risk = pred.get("risk_level", "Unknown")
                    risk_icon = "⚠️" if risk == "HIGH" else "✓"
                    st.metric("Risk Level", f"{risk_icon} {risk}")

                with col4:
                    model = pred.get("model", "unknown")
                    st.metric("Model", model.upper())

                # Additional info
                st.markdown("**Details:**")
                st.write(f"- **File:** {pred.get('filename', 'N/A')}")
                st.write(f"- **Timestamp:** {pred.get('timestamp', 'N/A')}")

                # Download option
                if st.button(
                    f"📥 Export Prediction #{len(history) - i}", key=f"export_{i}"
                ):
                    pred_json = json.dumps(pred, indent=2)
                    st.download_button(
                        "Download JSON",
                        data=pred_json,
                        file_name=f"prediction_{len(history) - i}.json",
                        mime="application/json",
                        key=f"download_{i}",
                    )

    else:
        # Table view
        df = pd.DataFrame(filtered_history)

        # Format columns
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.strftime(
                "%Y-%m-%d %H:%M"
            )

        if "confidence" in df.columns:
            df["confidence"] = df["confidence"].apply(lambda x: f"{x * 100:.2f}%")

        # Select columns to display
        display_cols = [
            "timestamp",
            "filename",
            "model",
            "prediction",
            "confidence",
            "risk_level",
        ]
        display_cols = [col for col in display_cols if col in df.columns]

        st.dataframe(df[display_cols], width="stretch", hide_index=True)

    st.divider()

    # Export options
    st.subheader("📥 Export History")

    col1, col2 = st.columns(2)

    with col1:
        # Export as JSON
        if st.button("📄 Export as JSON", width="stretch"):
            json_data = json.dumps(filtered_history, indent=2)
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                width="stretch",
            )

    with col2:
        # Export as CSV
        if st.button("📊 Export as CSV", width="stretch"):
            df = pd.DataFrame(filtered_history)
            csv_data = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                width="stretch",
            )

    # Clear history option
    st.divider()

    with st.expander("⚠️ Danger Zone"):
        st.warning("**Clear Prediction History**")
        st.write("This will permanently delete all prediction history.")

        if st.button("🗑️ Clear All History", type="secondary"):
            confirm = st.checkbox("I understand this action cannot be undone")
            if confirm:
                if st.button("✅ Confirm Delete", type="primary"):
                    # Clear history
                    with open(history_file, "w") as f:
                        json.dump([], f)
                    st.success("History cleared successfully!")
                    st.rerun()
