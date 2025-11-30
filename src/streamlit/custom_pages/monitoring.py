"""Monitoring tab - System and model monitoring."""

import streamlit as st
import pandas as pd
import plotly.express as px
import psutil
from pathlib import Path
import json

from src.config import get_settings

settings = get_settings()
MODELS_DIR = Path(settings.models_dir or "models")
DATA_DIR = Path(settings.processed_data_dir or "data/processed")


def render_monitoring_tab():
    """Render the monitoring dashboard tab."""

    st.header("🎛️ System Monitoring")

    # Model availability
    st.subheader("📦 Model Availability")

    models = ["cnn", "efficientnet_b3", "resnet50", "mobilenet_v3", "unet"]
    model_data = []

    for model_name in models:
        best_path = MODELS_DIR / f"best_{model_name}.keras"
        final_path = MODELS_DIR / f"{model_name}.keras"

        # Check if files exist (including symlinks)
        best_exists = best_path.exists()
        final_exists = final_path.exists()

        # Get size (follow symlinks to get actual file size)
        size_mb = 0
        try:
            if best_exists:
                # Use resolve() to follow symlink to actual file
                actual_path = best_path.resolve()
                if actual_path.exists():
                    size_mb = actual_path.stat().st_size / (1024 * 1024)
            elif final_exists:
                actual_path = final_path.resolve()
                if actual_path.exists():
                    size_mb = actual_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            st.warning(f"Could not read size for {model_name}: {e}")

        # Indicate if it's a symlink
        is_link = (
            best_path.is_symlink()
            if best_exists
            else final_path.is_symlink()
            if final_exists
            else False
        )
        link_indicator = " 🔗" if is_link else ""

        model_data.append(
            {
                "Model": model_name.upper(),
                "Best Model": f"✅{link_indicator}" if best_exists else "❌",
                "Final Model": f"✅{link_indicator}" if final_exists else "❌",
                "Size (MB)": f"{size_mb:.1f}" if size_mb > 0 else "N/A",
                "Status": "🟢 Ready" if (best_exists or final_exists) else "🔴 Missing",
            }
        )

    model_df = pd.DataFrame(model_data)
    st.dataframe(model_df, width="stretch", hide_index=True)

    # Storage info
    try:
        total_size = sum(
            f.stat().st_size
            for f in MODELS_DIR.glob("*.keras")
            if f.is_file() and not f.is_symlink()
        ) / (1024 * 1024)
    except Exception as e:
        total_size = 0
        st.warning(f"Could not calculate storage: {e}")

    st.metric("Total Model Storage", f"{total_size:.1f} MB")

    st.divider()

    # Prediction statistics
    st.subheader("📊 Prediction Statistics")

    history_file = DATA_DIR / "prediction_history.json"

    if history_file.exists():
        with open(history_file, "r") as f:
            history = json.load(f)

        if history:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Predictions", len(history))

            with col2:
                benign = sum(1 for h in history if h.get("prediction") == "Benign")
                st.metric("Benign", benign, f"{benign / len(history) * 100:.1f}%")

            with col3:
                malignant = sum(
                    1 for h in history if h.get("prediction") == "Malignant"
                )
                st.metric(
                    "Malignant", malignant, f"{malignant / len(history) * 100:.1f}%"
                )

            with col4:
                avg_conf = sum(h.get("confidence", 0) for h in history) / len(history)
                st.metric("Avg Confidence", f"{avg_conf * 100:.1f}%")

            # Predictions by model
            st.subheader("📈 Predictions by Model")

            model_counts = {}
            for h in history:
                model = h.get("model", "unknown")
                model_counts[model] = model_counts.get(model, 0) + 1

            if model_counts:
                model_df = pd.DataFrame(
                    list(model_counts.items()), columns=["Model", "Count"]
                )

                fig = px.pie(
                    model_df,
                    values="Count",
                    names="Model",
                    title="Distribution of Predictions by Model",
                )
                st.plotly_chart(fig, width="stretch")

            # Predictions over time
            st.subheader("📅 Predictions Timeline")

            history_df = pd.DataFrame(history)
            if "timestamp" in history_df.columns:
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
                history_df = history_df.sort_values("timestamp")

                # Daily predictions
                history_df["date"] = history_df["timestamp"].dt.date
                daily_counts = (
                    history_df.groupby("date").size().reset_index(name="count")
                )

                fig = px.line(
                    daily_counts,
                    x="date",
                    y="count",
                    title="Predictions per Day",
                    labels={"date": "Date", "count": "Number of Predictions"},
                )
                st.plotly_chart(fig, width="stretch")

                # Confidence distribution
                fig = px.histogram(
                    history_df,
                    x="confidence",
                    nbins=20,
                    title="Confidence Score Distribution",
                    labels={"confidence": "Confidence Score"},
                )
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No prediction history available")
    else:
        st.info("No predictions made yet. Make your first prediction!")

    st.divider()

    # System resources
    st.subheader("💻 System Resources")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**CPU Usage**")
        cpu_percent = psutil.cpu_percent(interval=1)
        st.metric("CPU", f"{cpu_percent}%")
        st.progress(cpu_percent / 100)

        # CPU details
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        st.caption(f"Cores: {cpu_count} | Freq: {cpu_freq.current:.0f} MHz")

    with col2:
        st.markdown("**Memory Usage**")
        memory = psutil.virtual_memory()
        st.metric("RAM", f"{memory.percent}%")
        st.progress(memory.percent / 100)

        # Memory details
        used_gb = memory.used / (1024**3)
        total_gb = memory.total / (1024**3)
        st.caption(f"Used: {used_gb:.1f} GB / {total_gb:.1f} GB")

    # Disk usage
    st.markdown("**Disk Usage**")
    disk = psutil.disk_usage("/")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total", f"{disk.total / (1024**3):.1f} GB")
    with col2:
        st.metric("Used", f"{disk.used / (1024**3):.1f} GB")
    with col3:
        st.metric("Free", f"{disk.free / (1024**3):.1f} GB")

    st.progress(disk.percent / 100)

    st.divider()

    # Application info
    st.subheader("ℹ️ Application Info")

    info_cols = st.columns(2)

    with info_cols[0]:
        st.markdown(
            """
        **Configuration:**
        - MLflow URI: `{}`
        - Models Dir: `{}`
        - Data Dir: `{}`
        """.format(
                settings.mlflow_tracking_uri,
                settings.models_dir,
                settings.processed_data_dir,
            )
        )

    with info_cols[1]:
        st.markdown("""
        **Endpoints:**
        - Streamlit: `http://localhost:8501`
        - MLflow UI: `http://localhost:5000`
        - API: `http://localhost:8000` (if enabled)
        """)

    # Refresh button
    st.divider()
    if st.button("🔄 Refresh Monitoring Data", width="stretch"):
        st.rerun()
