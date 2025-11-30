"""Metrics tab - Model performance comparison."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from mlflow.tracking import MlflowClient

from src.config import get_settings

settings = get_settings()


def load_model_metrics_from_mlflow():
    """Load metrics for all models from MLflow."""
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)

        if experiment is None:
            return None

        models = ["cnn", "efficientnet_b3", "resnet50", "mobilenet_v3", "unet"]
        all_metrics = {}

        for model_name in models:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"params.model_name = '{model_name}'",
                order_by=["metrics.val_accuracy DESC"],
                max_results=1,
            )

            if runs:
                run = runs[0]
                all_metrics[model_name] = {
                    "accuracy": run.data.metrics.get("val_accuracy", 0),
                    "precision": run.data.metrics.get("val_precision", 0),
                    "recall": run.data.metrics.get("val_recall", 0),
                    "f1": run.data.metrics.get("val_f1_score", 0),
                    "auc": run.data.metrics.get("val_auc", 0),
                    "loss": run.data.metrics.get("val_loss", 0),
                }

        return all_metrics if all_metrics else None

    except Exception as e:
        st.error(f"Error loading MLflow metrics: {e}")
        return None


def render_metrics_tab():
    """Render the model metrics comparison tab."""

    st.header("📊 Model Performance Comparison")

    # Load metrics
    with st.spinner("Loading metrics from MLflow..."):
        all_metrics = load_model_metrics_from_mlflow()

    if not all_metrics:
        st.warning("⚠️ No metrics found in MLflow")
        st.info("""
        **To see metrics here:**
        1. Train your models using `python -m src.models.training`
        2. Ensure MLflow tracking is enabled
        3. Refresh this page
        """)
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_metrics).T
    df.index.name = "Model"
    df = df.reset_index()
    df["Model"] = df["Model"].str.upper()

    # Overview metrics
    st.subheader("📈 Performance Overview")

    cols = st.columns(4)
    with cols[0]:
        best_accuracy = df["accuracy"].max()
        best_model = df.loc[df["accuracy"].idxmax(), "Model"]
        st.metric("Best Accuracy", f"{best_accuracy:.3f}", f"{best_model}")

    with cols[1]:
        avg_precision = df["precision"].mean()
        st.metric("Avg Precision", f"{avg_precision:.3f}")

    with cols[2]:
        avg_recall = df["recall"].mean()
        st.metric("Avg Recall", f"{avg_recall:.3f}")

    with cols[3]:
        best_auc = df["auc"].max()
        st.metric("Best AUC", f"{best_auc:.3f}")

    st.divider()

    # Detailed table
    st.subheader("📋 Detailed Metrics")

    styled_df = df.style.format(
        {
            "accuracy": "{:.4f}",
            "precision": "{:.4f}",
            "recall": "{:.4f}",
            "f1": "{:.4f}",
            "auc": "{:.4f}",
            "loss": "{:.4f}",
        }
    ).background_gradient(
        cmap="RdYlGn", subset=["accuracy", "precision", "recall", "f1", "auc"]
    )

    st.dataframe(styled_df, width="stretch", hide_index=True)

    st.divider()

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Accuracy Comparison")
        fig = px.bar(
            df,
            x="Model",
            y="accuracy",
            color="accuracy",
            color_continuous_scale="Viridis",
            text="accuracy",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("🎯 F1-Score Comparison")
        fig = px.bar(
            df, x="Model", y="f1", color="f1", color_continuous_scale="Blues", text="f1"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

    # Radar chart
    st.subheader("🕸️ Multi-Metric Radar Chart")

    fig = go.Figure()

    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "auc"]

    for _, row in df.iterrows():
        values = [row[m] for m in metrics_to_plot]
        values.append(values[0])  # Close the radar

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics_to_plot + [metrics_to_plot[0]],
                fill="toself",
                name=row["Model"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

    # Model ranking
    st.divider()
    st.subheader("🏆 Model Ranking")

    # Calculate overall score
    df["overall_score"] = (
        df["accuracy"] * 0.3
        + df["precision"] * 0.2
        + df["recall"] * 0.2
        + df["f1"] * 0.2
        + df["auc"] * 0.1
    )

    df_ranked = df.sort_values("overall_score", ascending=False)

    for i, (_, row) in enumerate(df_ranked.iterrows(), 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."

        with st.expander(f"{medal} {row['Model']} - Score: {row['overall_score']:.3f}"):
            cols = st.columns(5)
            cols[0].metric("Accuracy", f"{row['accuracy']:.3f}")
            cols[1].metric("Precision", f"{row['precision']:.3f}")
            cols[2].metric("Recall", f"{row['recall']:.3f}")
            cols[3].metric("F1", f"{row['f1']:.3f}")
            cols[4].metric("AUC", f"{row['auc']:.3f}")
