import numpy as np
from src.evaluation.metrics import ModelEvaluator


def test_compute_metrics_binary_with_proba():
    """Test the compute_metrics function with probability predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_pred_proba = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
    metrics = ModelEvaluator.compute_metrics(y_true, y_pred, y_pred_proba)
    assert "roc_auc" in metrics
    assert "logloss" in metrics
    assert metrics["accuracy"] == 1.0


def test_compute_metrics_roc_auc_exception(monkeypatch):
    """Force an exception in roc_auc_score to cover the except block."""
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    y_pred_proba = np.array([[0.5, 0.5], [0.5, 0.5]])

    import src.evaluation.metrics as metrics_module

    monkeypatch.setattr(
        metrics_module,
        "roc_auc_score",
        lambda a, b: (_ for _ in ()).throw(ValueError("bad")),
    )

    metrics = ModelEvaluator.compute_metrics(y_true, y_pred, y_pred_proba)
    assert "roc_auc" not in metrics


def test_get_confusion_matrix_with_one_hot():
    """Covers the one‑hot conversion in get_confusion_matrix."""
    y_true = np.array([[1, 0], [0, 1], [1, 0]])
    y_pred = np.array([[1, 0], [0, 1], [0, 1]])
    cm = ModelEvaluator.get_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_get_confusion_matrix_with_integers():
    """Covers direct path without one-hot"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm = ModelEvaluator.get_confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)


def test_get_roc_curve_triggers_argmax_y_true():
    """Covers the case where y_true is one‑hot encoded in get_roc_curve."""

    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])

    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])

    result = ModelEvaluator.get_roc_curve(y_true, y_pred_proba)

    assert isinstance(result, tuple)
    assert len(result) == 4

    auc_score = next(item for item in result if isinstance(item, float))
    assert auc_score == 1.0

    arrays = [item for item in result if isinstance(item, np.ndarray)]
    assert len(arrays) >= 2
    assert isinstance(auc_score, float)


def test_pr_one_hot_triggers_argmax():
    y_true_one_hot = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
    y_true_labels = np.array([0, 1, 0, 1])

    y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9]])

    p1, r1, ap1, th1 = ModelEvaluator.get_pr_curve(y_true_one_hot, y_pred_proba)
    p2, r2, ap2, th2 = ModelEvaluator.get_pr_curve(y_true_labels, y_pred_proba)

    assert np.allclose(p1, p2)
    assert np.allclose(r1, r2)
    assert ap1 == ap2 == 1.0


def test_class_distribution_argmax():
    y_true_one_hot = np.array([[1, 0], [0, 1], [1, 0], [1, 0]])
    y_true_labels = np.array([0, 1, 0, 0])

    dist1 = ModelEvaluator.get_class_distribution(y_true_one_hot)
    dist2 = ModelEvaluator.get_class_distribution(y_true_labels)

    assert dist1 == dist2
