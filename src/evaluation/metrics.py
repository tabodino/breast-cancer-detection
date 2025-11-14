import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    log_loss,
)
import tensorflow as tf
from loguru import logger


class ModelEvaluator:
    """Comprehensive model evaluation and metrics computation."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray = None
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels (one-hot or integer encoded)
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (for additional metrics)

        Returns:
            Dictionary of metrics
        """

        # Handle one-hot encoding
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

        # Per-class metrics
        metrics["classification_report"] = classification_report(
            y_true, y_pred, output_dict=True
        )

        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

        # AUC and ROC (for binary classification)
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
                y_pred_proba = y_pred_proba[:, 1]

            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
                metrics["logloss"] = float(log_loss(y_true, y_pred_proba))
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC: {e}")

        logger.info(
            f"Evaluation metrics: Acc={metrics['accuracy']:.4f}, "
            f"Prec={metrics['precision']:.4f}, "
            f"Rec={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}"
        )

        return metrics

    @staticmethod
    def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix."""
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)
        if len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)

        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple:
        """Get ROC curve data."""
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        return fpr, tpr, roc_auc, thresholds

    @staticmethod
    def get_pr_curve(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple:
        """Get Precision-Recall curve data."""
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
            y_pred_proba = y_pred_proba[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)

        return precision, recall, avg_precision, thresholds

    @staticmethod
    def get_class_distribution(y_true: np.ndarray) -> Dict:
        """Get class distribution in dataset."""
        if len(y_true.shape) > 1:
            y_true = np.argmax(y_true, axis=1)

        unique, counts = np.unique(y_true, return_counts=True)

        distribution = {}
        total = len(y_true)
        for cls, count in zip(unique, counts):
            distribution[f"class_{cls}"] = {
                "count": int(count),
                "percentage": float(count / total * 100),
            }

        return distribution


def evaluate_model_on_dataset(
    model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray
) -> Dict:
    """
    Evaluate a TensorFlow model on test dataset.
    """

    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Compute metrics
    evaluator = ModelEvaluator()
    metrics = evaluator.compute_metrics(y_test, y_pred, y_pred_proba)

    return metrics


if __name__ == "__main__":
    logger.info("Metrics module ready for use")
