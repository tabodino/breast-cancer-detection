"""
Comprehensive tests for evaluation metrics and visualization.
Covers metrics computation, visualization generation, and result analysis.
Includes tests for one-hot encoded labels.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
from src.evaluation.metrics import ModelEvaluator, evaluate_model_on_dataset
from src.evaluation.visualization import MetricsVisualizer


class TestModelEvaluator:
    """Tests for ModelEvaluator metrics computation."""

    @pytest.fixture
    def binary_classification_data(self):
        """Create binary classification test data."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        y_pred_proba = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.9, 0.1],
                [0.3, 0.7],
                [0.2, 0.8],
            ]
        )
        return y_true, y_pred, y_pred_proba

    @pytest.fixture
    def binary_classification_data_onehot(self):
        """Create binary classification test data with one-hot encoding."""
        y_true = np.array(
            [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]]
        )
        y_pred = np.array(
            [[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
        )
        y_pred_proba = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.9, 0.1],
                [0.3, 0.7],
                [0.2, 0.8],
            ]
        )
        return y_true, y_pred, y_pred_proba

    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification test data."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 2])
        return y_true, y_pred, None

    @pytest.fixture
    def multiclass_data_onehot(self):
        """Create multiclass classification test data with one-hot encoding."""
        y_true = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        y_pred = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
        )
        return y_true, y_pred, None

    def test_compute_metrics_binary(self, binary_classification_data):
        """compute_metrics should work for binary classification."""
        y_true, y_pred, y_pred_proba = binary_classification_data

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred, y_pred_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" in metrics

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1

    def test_compute_metrics_multiclass(self, multiclass_data):
        """compute_metrics should work for multiclass."""
        y_true, y_pred, _ = multiclass_data

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0

    def test_compute_metrics_onehot(self):
        """compute_metrics should handle one-hot encoded labels."""
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_pred = np.array([[1, 0], [0, 1], [1, 0]])

        metrics = ModelEvaluator.compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0

    def test_confusion_matrix(self, binary_classification_data):
        """get_confusion_matrix should compute correctly."""
        y_true, y_pred, _ = binary_classification_data

        cm = ModelEvaluator.get_confusion_matrix(y_true, y_pred)

        assert cm.shape == (2, 2)
        assert cm[0, 0] + cm[0, 1] + cm[1, 0] + cm[1, 1] == len(y_true)

    def test_roc_curve(self, binary_classification_data):
        """get_roc_curve should compute correctly."""
        y_true, _, y_pred_proba = binary_classification_data

        fpr, tpr, roc_auc, _ = ModelEvaluator.get_roc_curve(y_true, y_pred_proba)

        assert len(fpr) > 0
        assert len(tpr) > 0
        assert 0 <= roc_auc <= 1

    def test_pr_curve(self, binary_classification_data):
        """get_pr_curve should compute correctly."""
        y_true, _, y_pred_proba = binary_classification_data

        precision, recall, avg_precision, _ = ModelEvaluator.get_pr_curve(
            y_true, y_pred_proba
        )

        assert len(precision) > 0
        assert len(recall) > 0
        assert 0 <= avg_precision <= 1

    def test_class_distribution(self, binary_classification_data):
        """get_class_distribution should compute correctly."""
        y_true, _, _ = binary_classification_data

        dist = ModelEvaluator.get_class_distribution(y_true)

        assert "class_0" in dist
        assert "class_1" in dist
        assert dist["class_0"]["count"] + dist["class_1"]["count"] == len(y_true)
        assert 0 <= dist["class_0"]["percentage"] <= 100


class TestMetricsVisualizer:
    """Tests for MetricsVisualizer visualization generation."""

    @pytest.fixture
    def dummy_metrics_data(self):
        """Create dummy data for visualization."""
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 0, 1, 1])
        y_pred_proba = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.9, 0.1],
                [0.3, 0.7],
                [0.2, 0.8],
            ]
        )

        history = {
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.55, 0.45, 0.35],
            "accuracy": [0.7, 0.8, 0.85],
            "val_accuracy": [0.65, 0.75, 0.80],
        }

        return y_true, y_pred, y_pred_proba, history

    @pytest.fixture
    def dummy_metrics_data_onehot(self):
        """Create dummy data for visualization with one-hot encoding."""
        y_true = np.array(
            [[1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [0, 1]]
        )
        y_pred = np.array(
            [[1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]
        )
        y_pred_proba = np.array(
            [
                [0.9, 0.1],
                [0.1, 0.9],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.7, 0.3],
                [0.9, 0.1],
                [0.3, 0.7],
                [0.2, 0.8],
            ]
        )

        history = {
            "loss": [0.5, 0.4, 0.3],
            "val_loss": [0.55, 0.45, 0.35],
            "accuracy": [0.7, 0.8, 0.85],
            "val_accuracy": [0.65, 0.75, 0.80],
        }

        return y_true, y_pred, y_pred_proba, history

    def test_visualizer_initialization(self):
        """MetricsVisualizer should initialize correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            assert viz.output_dir.exists()

    def test_plot_confusion_matrix(self, dummy_metrics_data):
        """plot_confusion_matrix should save file."""
        y_true, y_pred, _, _ = dummy_metrics_data

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_confusion_matrix(y_true, y_pred)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_confusion_matrix_onehot(self, dummy_metrics_data_onehot):
        """plot_confusion_matrix should handle one-hot encoded labels."""
        y_true, y_pred, _, _ = dummy_metrics_data_onehot

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_confusion_matrix(y_true, y_pred)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_roc_curve(self, dummy_metrics_data):
        """plot_roc_curve should save file."""
        y_true, _, y_pred_proba, _ = dummy_metrics_data

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_roc_curve(y_true, y_pred_proba)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_roc_curve_onehot(self, dummy_metrics_data_onehot):
        """plot_roc_curve should handle one-hot encoded labels."""
        y_true, _, y_pred_proba, _ = dummy_metrics_data_onehot

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_roc_curve(y_true, y_pred_proba)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_pr_curve(self, dummy_metrics_data):
        """plot_pr_curve should save file."""
        y_true, _, y_pred_proba, _ = dummy_metrics_data

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_pr_curve(y_true, y_pred_proba)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_pr_curve_onehot(self, dummy_metrics_data_onehot):
        """plot_pr_curve should handle one-hot encoded labels."""
        y_true, _, y_pred_proba, _ = dummy_metrics_data_onehot

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_pr_curve(y_true, y_pred_proba)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_training_history(self, dummy_metrics_data):
        """plot_training_history should save file."""
        _, _, _, history = dummy_metrics_data

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_training_history(history)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_class_distribution(self, dummy_metrics_data):
        """plot_class_distribution should save file."""
        y_true, _, _, _ = dummy_metrics_data

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_class_distribution(y_true)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_class_distribution_onehot(self, dummy_metrics_data_onehot):
        """plot_class_distribution should handle one-hot encoded labels."""
        y_true, _, _, _ = dummy_metrics_data_onehot

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_class_distribution(y_true)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_metrics_comparison(self):
        """plot_metrics_comparison should save file."""
        results = {
            "model1": {"accuracy": 0.85, "precision": 0.87, "recall": 0.83, "f1": 0.85},
            "model2": {"accuracy": 0.90, "precision": 0.92, "recall": 0.88, "f1": 0.90},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_metrics_comparison(results)

            assert Path(save_path).exists()
            plt.close("all")

    def test_plot_confusion_matrix_multiclass_onehot(self):
        """plot_confusion_matrix should handle multiclass one-hot encoded labels."""
        y_true = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        y_pred = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_confusion_matrix(
                y_true, y_pred, class_names=["Class 0", "Class 1", "Class 2"]
            )

            assert Path(save_path).exists()
            plt.close("all")


class TestOneHotEncodingEdgeCases:
    """Specific tests for one-hot encoding edge cases."""

    def test_confusion_matrix_with_both_onehot(self):
        """Test confusion matrix when both y_true and y_pred are one-hot."""
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_pred = np.array([[1, 0], [1, 0], [1, 0], [0, 1]])

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_confusion_matrix(y_true, y_pred)

            assert Path(save_path).exists()
            plt.close("all")

    def test_roc_curve_with_onehot_y_true(self):
        """Test ROC curve when y_true is one-hot encoded."""
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_roc_curve(y_true, y_pred_proba)

            assert Path(save_path).exists()
            plt.close("all")

    def test_pr_curve_with_onehot_y_true(self):
        """Test PR curve when y_true is one-hot encoded."""
        y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.3, 0.7]])

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_pr_curve(y_true, y_pred_proba)

            assert Path(save_path).exists()
            plt.close("all")

    def test_class_distribution_with_onehot(self):
        """Test class distribution when y is one-hot encoded."""
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 0], [0, 1, 0]])

        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MetricsVisualizer(output_dir=Path(tmpdir))
            save_path = viz.plot_class_distribution(y)

            assert Path(save_path).exists()
            plt.close("all")


class TestEvaluationPipeline:
    """Integration tests for evaluation pipeline."""

    def test_evaluate_model_on_dataset(self):
        """evaluate_model_on_dataset should compute all metrics."""
        import tensorflow as tf

        # Create simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(10, input_shape=(5,), activation="relu"),
                tf.keras.layers.Dense(2, activation="softmax"),
            ]
        )
        model.compile(optimizer="adam", loss="categorical_crossentropy")

        # Dummy data
        X_test = np.random.randn(10, 5)
        y_test = tf.keras.utils.to_categorical([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], 2)

        metrics = evaluate_model_on_dataset(model, X_test, y_test)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
