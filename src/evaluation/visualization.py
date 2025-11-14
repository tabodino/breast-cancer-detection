import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from src.logger import logger
from src.config import get_settings


settings = get_settings()


class MetricsVisualizer:
    """Create visualizations for model evaluation."""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or settings.artifacts_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        save_name: str = "confusion_matrix.png",
    ) -> str:
        """Plot confusion matrix."""

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(cm))]

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            cbar_kws={"label": "Count"},
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path}")
        return str(save_path)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_name: str = "roc_curve.png",
    ) -> str:
        """Plot ROC curve."""

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax.plot(
            [0, 1],
            [0, 1],
            color="navy",
            lw=2,
            linestyle="--",
            label="Random Classifier",
        )
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"ROC curve saved to {save_path}")
        return str(save_path)

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        save_name: str = "pr_curve.png",
    ) -> str:
        """Plot Precision-Recall curve."""

        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(recall, precision, lw=2, label="Precision-Recall curve")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"PR curve saved to {save_path}")
        return str(save_path)

    def plot_training_history(
        self, history: Dict, save_name: str = "training_history.png"
    ) -> str:
        """Plot training history (loss and accuracy)."""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(history.get("loss", []), label="Training Loss", linewidth=2)
        ax1.plot(history.get("val_loss", []), label="Validation Loss", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Model Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(history.get("accuracy", []), label="Training Accuracy", linewidth=2)
        ax2.plot(
            history.get("val_accuracy", []), label="Validation Accuracy", linewidth=2
        )
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Model Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Training history saved to {save_path}")
        return str(save_path)

    def plot_class_distribution(
        self,
        y: np.ndarray,
        class_names: List[str] = None,
        save_name: str = "class_distribution.png",
    ) -> str:
        """Plot class distribution in dataset."""

        if len(y.shape) > 1:
            y = np.argmax(y, axis=1)

        unique, counts = np.unique(y, return_counts=True)

        if class_names is None:
            class_names = [f"Class {i}" for i in unique]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("husl", len(unique))
        ax.bar(class_names, counts, color=colors, edgecolor="black", linewidth=1.5)
        ax.set_ylabel("Count")
        ax.set_title("Class Distribution")
        ax.grid(True, alpha=0.3, axis="y")

        # Add count labels on bars
        for i, (name, count) in enumerate(zip(class_names, counts)):
            ax.text(i, count, str(count), ha="center", va="bottom", fontweight="bold")

        fig.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Class distribution saved to {save_path}")
        return str(save_path)

    def plot_metrics_comparison(
        self, results: Dict[str, Dict], save_name: str = "metrics_comparison.png"
    ) -> str:
        """Compare metrics across multiple models."""

        models = list(results.keys())
        metrics = ["accuracy", "precision", "recall", "f1"]

        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(models))
        width = 0.2

        for i, metric in enumerate(metrics):
            values = [results[model].get(metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=metric.capitalize())

        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim([0, 1.0])

        fig.tight_layout()
        save_path = self.output_dir / save_name
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Metrics comparison saved to {save_path}")
        return str(save_path)


if __name__ == "__main__":
    logger.info("Visualization module ready for use")
