"""MLflow History tab - Experiment tracking visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from mlflow.tracking import MlflowClient
from loguru import logger

from src.config import get_settings

settings = get_settings()


def render_mlflow_tab():
    """Render the MLflow experiment history tab."""

    st.header("📈 MLflow Experiment Tracking")

    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)

        if experiment is None:
            st.warning("⚠️ No MLflow experiment found")
            st.info(f"Expected experiment name: `{settings.mlflow_experiment_name}`")

            if st.button("Create Experiment"):
                import mlflow

                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                mlflow.create_experiment(settings.mlflow_experiment_name)
                st.success("✅ Experiment created!")
                st.rerun()
            return

        # Experiment info
        st.subheader("🔬 Experiment Details")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Experiment Name", experiment.name)
        with col2:
            st.metric("Experiment ID", experiment.experiment_id)
        with col3:
            st.metric("Artifact Location", "mlruns/")

        st.divider()

        # Get all runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=100,
        )

        if not runs:
            st.info("No runs found. Train a model to see results here.")
            return

        # Create runs dataframe
        runs_data = []
        for run in runs:
            runs_data.append(
                {
                    "Run ID": run.info.run_id[:8],
                    "Full Run ID": run.info.run_id,
                    "Model": run.data.params.get("model_name", "N/A"),
                    "Accuracy": run.data.metrics.get("val_accuracy", 0),
                    "Precision": run.data.metrics.get("val_precision", 0),
                    "Recall": run.data.metrics.get("val_recall", 0),
                    "F1": run.data.metrics.get("val_f1_score", 0),
                    "Loss": run.data.metrics.get("val_loss", 0),
                    "Epochs": run.data.params.get("epochs", "N/A"),
                    "Batch Size": run.data.params.get("batch_size", "N/A"),
                    "Status": run.info.status,
                    "Duration": f"{(run.info.end_time - run.info.start_time) / 1000 / 60:.1f} min"
                    if run.info.end_time
                    else "N/A",
                    "Start Time": datetime.fromtimestamp(
                        run.info.start_time / 1000
                    ).strftime("%Y-%m-%d %H:%M"),
                }
            )

        df = pd.DataFrame(runs_data)

        # Summary metrics
        st.subheader("📊 Training Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", len(df))
        with col2:
            completed = len(df[df["Status"] == "FINISHED"])
            st.metric("Completed", completed)
        with col3:
            best_acc = df["Accuracy"].max()
            st.metric("Best Accuracy", f"{best_acc:.3f}")
        with col4:
            models_trained = df["Model"].nunique()
            st.metric("Models Trained", models_trained)

        st.divider()

        # Filters
        st.subheader("🔍 Filter Runs")

        col1, col2, col3 = st.columns(3)

        with col1:
            model_filter = st.multiselect(
                "Filter by Model",
                options=df["Model"].unique(),
                default=df["Model"].unique(),
            )

        with col2:
            status_filter = st.multiselect(
                "Filter by Status",
                options=df["Status"].unique(),
                default=df["Status"].unique(),
            )

        with col3:
            min_accuracy = st.slider(
                "Min Accuracy", min_value=0.0, max_value=1.0, value=0.0, step=0.05
            )

        # Apply filters
        filtered_df = df[
            (df["Model"].isin(model_filter))
            & (df["Status"].isin(status_filter))
            & (df["Accuracy"] >= min_accuracy)
        ]

        st.write(f"**Showing {len(filtered_df)} of {len(df)} runs**")

        # Display table
        display_cols = [
            "Run ID",
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Loss",
            "Status",
            "Start Time",
        ]
        st.dataframe(filtered_df[display_cols], width="stretch", hide_index=True)

        st.divider()

        # Run details
        st.subheader("📋 Run Details")

        selected_run_id = st.selectbox(
            "Select a run to view details",
            options=filtered_df["Full Run ID"].tolist(),
            format_func=lambda x: f"{df[df['Full Run ID'] == x]['Model'].iloc[0]} - {x[:8]}",
        )

        if selected_run_id:
            run = client.get_run(selected_run_id)

            # Run info
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Parameters:**")
                params_df = pd.DataFrame(
                    list(run.data.params.items()), columns=["Parameter", "Value"]
                )
                st.dataframe(params_df, hide_index=True, width="stretch")

            with col2:
                st.markdown("**Metrics:**")
                metrics_df = pd.DataFrame(
                    list(run.data.metrics.items()), columns=["Metric", "Value"]
                )
                st.dataframe(metrics_df, hide_index=True, width="stretch")

            # Metric history
            st.markdown("**Training Progress:**")

            metric_to_plot = st.selectbox(
                "Select metric",
                ["val_accuracy", "val_loss", "val_precision", "val_recall"],
            )

            try:
                metric_history = client.get_metric_history(
                    selected_run_id, metric_to_plot
                )

                if metric_history:
                    history_df = pd.DataFrame(
                        [{"Epoch": m.step, "Value": m.value} for m in metric_history]
                    )

                    fig = px.line(
                        history_df,
                        x="Epoch",
                        y="Value",
                        title=f"{metric_to_plot} over Epochs",
                    )
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.info("No metric history available for this run")
            except Exception as e:
                st.warning("Could not load metric history")
                logger.warning(f"Could not load metric history: {e}")

        st.divider()

        # MLflow UI link
        st.info(f"""
        🔗 **Access full MLflow UI:**
        
        Run in terminal:
        ```bash
        mlflow ui --backend-store-uri {settings.mlflow_tracking_uri}
        ```
        
        Then open: http://localhost:5000
        """)

    except Exception as e:
        st.error(f"Error loading MLflow data: {e}")
        st.exception(e)
